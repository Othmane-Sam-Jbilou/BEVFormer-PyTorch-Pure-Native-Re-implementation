import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def project_3d_to_2d(points_3d, intrinsics, extrinsics_R: Tensor, extrinsics_T: Tensor, img_hw):
    """
        Projects 3D points in ego space to 2D normalized pixel coordinates for camera views.

        Args:
            points_3d: Tensor of shape (N_points, 3) representing (x, y, z) in ego coordinates.
            intrinsics: Tensor of shape (N_cam, 3, 3) camera intrinsic matrices K.
            extrinsics_R: Tensor of shape (N_cam, 3, 3) rotation matrices (ego -> camera).
            extrinsics_T: Tensor of shape (N_cam, 3, 1) translation vectors (ego -> camera).
            img_hw: Tuple of (height, width) of image feature maps.

        Returns:
            uv_normalized: Tensor of shape (N_cam, N_points, 2) in range [-1, 1].
                        Convention is F.grid_sample(..., align_corners=False), which is
                        grid_sample's default. Pass align_corners=False when sampling.
            valid_mask: Boolean Tensor of shape (N_cam, N_points) indicating if points
                        fall in front of the camera (depth > 0) and inside image bounds.
    """

    N_cam = 6
    H, W = img_hw

    # 1. Transform points from ego frame to camera frame
    # P_cam = R * P_ego + t   (row-vector form: P_ego @ R^T + t)

    points_3d = points_3d.unsqueeze(0).unsqueeze(0)

    points_cam = points_3d @ extrinsics_R.transpose(-1, -2) + extrinsics_T.squeeze(-1).unsqueeze(2)                             # (N_cam, N_points, 3)

    depths = points_cam[..., 2]

    # 2. Project onto image plane using intrinsics
    # [u*z, v*z, z]^T = K * P_cam
    points_2d_homo = points_cam @ intrinsics.transpose(-1, -2)                       # (N_cam, N_points, 3)

    depths_safe = torch.clamp(depths, min=1e-5)
    u = points_2d_homo[..., 0] / depths_safe
    v = points_2d_homo[..., 1] / depths_safe

    valid_mask = (depths > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # 3. Pixel centre u in [0, W-1] -> [-1, 1], matching grid_sample(align_corners=False):
    #    x_norm = (2*u + 1) / W - 1
    u_norm = (2.0 * u + 1.0) / W - 1.0
    v_norm = (2.0 * v + 1.0) / H - 1.0

    uv_normalized = torch.stack([u_norm, v_norm], dim=-1)                          # (N_cam, N_points, 2)

    return uv_normalized, valid_mask

class Backbone(nn.Module):
    def __init__(self, output_channels , freeze_backbone=True):
        super().__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.freeze_backbone = freeze_backbone

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        if freeze_backbone:
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False

        self.proj = nn.Conv2d(256, output_channels, kernel_size=1)

    def forward(self, x):

        if self.freeze_backbone:
            with torch.no_grad():
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                out = self.proj(x)
        else:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            out = self.proj(x)

        return out

class DeformableAttention(nn.Module):

    def __init__(self, emb_dim, n_head, num_points=4, n_levels=1):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_head = n_head
        self.num_points = num_points
        self.n_levels = n_levels
        self.head_dim = emb_dim // n_head

        # Per-head, per-level, per-point sampling offsets (2D)
        self.sampling_offsets = nn.Linear(
            emb_dim, n_head * n_levels * num_points * 2
        )

        self.attention_weights = nn.Linear(
            emb_dim, n_head * n_levels * num_points
        )

        self.value_proj = nn.Linear(emb_dim, emb_dim)
        self.output_proj = nn.Linear(emb_dim, emb_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Zero-init offsets so training starts from identity sampling
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.0)

        # Small init for attention weights (uniform across points at start)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)

    def forward(self, query, value, reference_points, spatial_shape=(200,200)):

        B, Nq, C = query.shape
        Nk = value.shape[1]
        Hk, Wk = spatial_shape

        #Value projection and reshape to heads
        value = self.value_proj(value)                       # [B, Nk, C]
        value = value.view(B, Nk, self.n_head, self.head_dim)
        value = value.permute(0, 2, 1, 3).contiguous()       # [B, H, Nk, D]

        offsets = self.sampling_offsets(query)
        offsets = offsets.view(B, Nq, self.n_head, self.n_levels, self.num_points, 2)

        # Normalize offsets so a unit offset = one feature-map cell.
        offsets = offsets / offsets.new_tensor([Wk, Hk]).view(1, 1, 1, 1, 1, 2)

        # attn_weights: [B, Nq, H, L, P] -> softmax over (L, P)
        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(B, Nq, self.n_head, self.n_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.view(B, Nq, self.n_head, self.n_levels, self.num_points
)

        #Compute sampling locations in normalized coords
        # reference_points: [B, Nq, 2] -> [B, Nq, 1, 1, 1, 2]
        ref = reference_points.view(B, Nq, 1, 1, 1, 2)
        # sampling_locations: [B, Nq, H, L, P, 2], in [0, 1]
        sampling_locations = ref + offsets
        # Clamp
        sampling_locations = sampling_locations.clamp(0.0, 1.0)

        # Reshape for grid_sample
        L, P = self.n_levels, self.num_points
        value = value.view(B, self.n_head, self.head_dim, Hk, Wk)

        # sampling_locations: [B, Nq, H, L, P, 2] -> [B*H*L, Nq*P, 1, 2]
        # Move H to batch: [B, H, Nq, L, P, 2]
        sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4, 5)

        # Convert to [-1, 1] for grid_sample
        sampling_locations = 2.0 * sampling_locations - 1.0

        value = value.reshape(B * self.n_head, self.head_dim, Hk, Wk)

        # sampling_locations for grid_sample: [B*H, Nq, L*P, 2] -> [B*H, Nq, L*P, 2]
        sampling_locations = sampling_locations.reshape(B * self.n_head, Nq, L * P, 2)

        sampled = F.grid_sample(
            value,
            sampling_locations,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )  # [B*H, D, Nq, L*P]

        # Weighted sum over (L, P)
        # attn_weights: [B, Nq, H, L, P] -> [B*H, Nq, L*P]
        attn_weights = attn_weights.reshape(B, Nq, self.n_head, L * P)
        attn_weights = attn_weights.permute(0, 2, 1, 3).reshape(B * self.n_head, Nq, L * P)  # [B*H, Nq, L*P]

        # sampled: [B*H, D, Nq, L*P] -> [B*H, Nq, D, L*P]
        sampled = sampled.permute(0, 2, 1, 3)

        # Weighted sum over the L*P axis
        out = (sampled * attn_weights.unsqueeze(2)).sum(dim=-1)  # [B*H, Nq, D]
        out = out.view(B, self.n_head, Nq, self.head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Nq, C)

        return self.output_proj(out)

class TemporalSelfAttention(nn.Module):
    def __init__(self, emb_dim, n_head,num_points=4, bev_range=(-51.2, 51.2, -51.2, 51.2), spatial_shape=(200,200)):
        super().__init__()

        self.emb_dim = emb_dim
        self.bev_range = bev_range
        self.self_attn = DeformableAttention(emb_dim=emb_dim, n_head=n_head, num_points=num_points, n_levels=1)

        H, W = spatial_shape
        x_min, x_max, y_min, y_max = bev_range
        x = torch.linspace(x_min, x_max, W)
        y = torch.linspace(y_min, y_max, H)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        zz = torch.zeros_like(xx)
        grid_3d = torch.stack(
            [xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1
        )
        self.register_buffer("grid_3d", grid_3d, persistent=False)

        # Normalized reference points for queries
        ys = torch.linspace(0.0, 1.0, H)
        xs = torch.linspace(0.0, 1.0, W)
        yy_ref, xx_ref = torch.meshgrid(ys, xs, indexing="ij")
        ref = torch.stack([xx_ref.reshape(-1), yy_ref.reshape(-1)], dim=-1)
        self.register_buffer("ref_points", ref, persistent=False)

    def warp_bev(self, prev_bev_feat, R_curr2prev, t_curr2prev):
        #warp bev to align the 2 timeframes
        B, H, W, C = prev_bev_feat.shape
        x_min, x_max, y_min, y_max = self.bev_range

        grid_3d = self.grid_3d.unsqueeze(0).expand(B, -1, -1)

        grid_prev = (
            torch.bmm(grid_3d, R_curr2prev.transpose(1, 2))
            + t_curr2prev.transpose(1, 2)
        )

        u_prev, v_prev = grid_prev[:, :, 0], grid_prev[:, :, 1]
        u_norm = (2.0 * (u_prev - x_min) / (x_max - x_min)) - 1.0
        v_norm = (2.0 * (v_prev - y_min) / (y_max - y_min)) - 1.0

        grid_uv = torch.stack([u_norm, v_norm], dim=-1).view(B, H, W, 2)
        feat = prev_bev_feat.permute(0, 3, 1, 2).contiguous()
        return F.grid_sample(
            feat,
            grid_uv,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def forward(self, prev_bev_feat, curr_bev_query, has_prev, R_curr2prev, t_curr2prev):
        B, H, W, C = curr_bev_query.shape
        curr_flat = curr_bev_query.view(B, H * W, C)

        # use the save query as value if there's no previous timestamp
        if (has_prev is not None and not has_prev.any()) or prev_bev_feat is None:
            attn_out = self.self_attn(
                query=curr_flat,
                value=curr_flat,
                reference_points=self.ref_points.unsqueeze(0).expand(B, -1, -1),
                spatial_shape=(H, W),
            )
            fused_bev = curr_flat + attn_out
            return fused_bev.view(B, H, W, C)

        warped_prev_bev = self.warp_bev(prev_bev_feat, R_curr2prev, t_curr2prev)

        # Stack along the Batch dimension to preserve the HxW spatial grid
        value_stacked = torch.cat([curr_bev_query.permute(0, 3, 1, 2), warped_prev_bev], dim=0)  # [2B, C, H, W]
        value_flat = value_stacked.flatten(2).transpose(1, 2)  # [2B, H*W, C]

        # Duplicate queries and reference points for the 2B batch size
        query_dup = curr_flat.repeat(2, 1, 1)  # [2B, H*W, C]
        ref_dup = self.ref_points.unsqueeze(0).expand(2 * B, -1, -1)

        attn_out_stacked = self.self_attn(
            query=query_dup,
            value=value_flat,
            reference_points=ref_dup,
            spatial_shape=(H, W),
        )

        # Average the attention outputs back to batch size B
        attn_curr, attn_prev = attn_out_stacked.chunk(2, dim=0)
        attn_out = (attn_curr + attn_prev) / 2.0
        fused_bev = curr_flat + attn_out

        return fused_bev.view(B, H, W, C)

class SpatialCrossAttention(nn.Module):
    def __init__(self, emb_dim, num_cams=6, num_z_anchors=4):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_cams = num_cams
        self.num_z_anchors = num_z_anchors

        self.attention_weights = nn.Linear(emb_dim, num_cams * num_z_anchors)

        self.output_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, bev_queries, projection, img_features, K, R, t):
        B, H_bev, W_bev, C = bev_queries.shape
        N_bev = H_bev * W_bev
        _, _, _, H_img, W_img = img_features.shape

        uv_norm, valid_mask = projection

        weights = torch.sigmoid(self.attention_weights(bev_queries.view(B, N_bev, C)))  # [B, N_bev, C]
        weights = weights.view(B, N_bev, self.num_cams, self.num_z_anchors)  # [B, N_bev, 6, 4]

        aggregated = torch.zeros(B, N_bev, C, device=bev_queries.device)


        for cam_idx in range(self.num_cams):
            cam_feat = img_features[:, cam_idx, ...] #(B, 1, C, H, W)

            cam_uv = uv_norm[:, cam_idx].unsqueeze(1)  # (B, 1, N_points, 2)

            sampled = F.grid_sample(cam_feat, cam_uv, align_corners=True, padding_mode='zeros')
            sampled = sampled.squeeze(2).transpose(1, 2)
            mask = valid_mask[:, cam_idx].unsqueeze(-1)
            sampled = sampled.masked_fill(~mask, 0.0)

            sampled = sampled.view(B, N_bev, self.num_z_anchors, C) #(B, N_bev, Z, C)
            w = weights[:, :, cam_idx, :].unsqueeze(-1) #(B, N_bev, Z, 1)

            aggregated = aggregated + (sampled * w).sum(dim=2) #(B, N_BEV, C) sum over the anchors

        updated_queries = bev_queries + self.output_proj(aggregated.view(B, H_bev, W_bev, C))
        return updated_queries

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, n_cams, num_z_anchors, n_head, bev_range=(-51.2, 51.2, -51.2, 51.2), dropout=0.1):
        super().__init__()

        self.TSA = TemporalSelfAttention(emb_dim=emb_dim, n_head=n_head, bev_range=bev_range)
        self.SCA = SpatialCrossAttention(emb_dim=emb_dim, num_cams=n_cams, num_z_anchors=num_z_anchors)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, history_BEV, BEV_queries, projection, img_features, has_prev, R_curr2prev, t_curr2prev, K, R, t):

        tsa_out = self.TSA(history_BEV, BEV_queries, has_prev, R_curr2prev, t_curr2prev)
        BEV_queries = BEV_queries + self.dropout1(tsa_out)
        BEV_queries = self.norm1(BEV_queries)

        sca_out = self.SCA(BEV_queries, projection, img_features, K, R, t)
        BEV_queries = BEV_queries + self.dropout2(sca_out)
        BEV_queries = self.norm2(BEV_queries)

        ffn_out = self.ffn(BEV_queries)
        BEV_queries = BEV_queries + self.dropout3(ffn_out)
        BEV_queries = self.norm3(BEV_queries)

        return BEV_queries

class BEVformerEncoder(nn.Module):
    def __init__(self, emb_dim, n_cams, num_z_anchors, n_head, n_enc, spatial_shape=(200,200),z_bounds=(-5.0, 3.0), bev_range=(-51.2, 51.2, -51.2, 51.2)):
        super().__init__()
        self.bev_h , self.bev_w = spatial_shape
        self.N_bev = self.bev_h * self.bev_w
        self.num_z_anchors = num_z_anchors
        self.bev_pos = nn.Parameter(torch.zeros(self.bev_h , self.bev_w, emb_dim))
        nn.init.trunc_normal_(self.bev_pos, std=0.02)

        def generate_bev_3d_coordinates():
            x_min, x_max, y_min, y_max = bev_range
            H_bev, W_bev = spatial_shape
            z_min, z_max = z_bounds

            x_coords = torch.linspace(x_min, x_max, W_bev)
            y_coords = torch.linspace(y_min, y_max, H_bev)
            z_coords = torch.linspace(z_min, z_max, num_z_anchors)

            # Create a 3D coordinate meshgrid: shape (H_bev, W_bev, num_z_anchors)
            yy, xx, zz = torch.meshgrid(y_coords, x_coords, z_coords, indexing='ij')

            # Stack into (X, Y, Z) channels: shape (H_bev, W_bev, num_z_anchors, 3)
            coords_3d = torch.stack([xx, yy, zz], dim=-1)

            bev_coords_3d = coords_3d.view(H_bev * W_bev, num_z_anchors, 3)

            return bev_coords_3d

        bev_coords_3d = generate_bev_3d_coordinates()
        self.register_buffer('bev_coords_3d', bev_coords_3d)

        self.Layers = nn.ModuleList([
            EncoderLayer(emb_dim=emb_dim, n_cams=n_cams, num_z_anchors=num_z_anchors, n_head=n_head) for _ in range(n_enc)
        ])

    def forward(self, history_BEV, BEV_queries, img_features, has_prev, R_curr2prev, t_curr2prev, K, R, t):
        B = BEV_queries.shape[0]
        BEV_queries = BEV_queries + self.bev_pos.unsqueeze(0)

        flat_coords_3d = self.bev_coords_3d.reshape(self.N_bev * self.num_z_anchors, 3)
        projection= project_3d_to_2d(flat_coords_3d, K, R, t, (self.bev_h , self.bev_w))

        for layer in self.Layers:
            BEV_queries = layer(history_BEV, BEV_queries, projection, img_features, has_prev, R_curr2prev, t_curr2prev, K, R, t)

        return BEV_queries

class BEVSegmentationHead(nn.Module):

    #small CNN as a decoder to generate the segmentation maps
    def __init__(self, in_channels, num_classes=5, hidden_dim=128):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, bev_tokens):

        bev_grid = bev_tokens.permute(0 ,3 ,1 ,2)
        feat = self.decoder(bev_grid)
        logits = self.classifier(feat)

        return logits

class BEVformer(nn.Module):
    def __init__(self, emb_dim=256, n_cams=6, num_z_anchors=4, n_head=8, n_enc=6, freeze_backbone=True):
        super().__init__()

        self.emb_dim = emb_dim
        self.backbone = Backbone(output_channels=emb_dim, freeze_backbone=freeze_backbone)
        self.encoder = BEVformerEncoder(emb_dim=emb_dim, n_cams=n_cams, num_z_anchors=num_z_anchors, n_head=n_head, n_enc=n_enc)
        self.decoder = BEVSegmentationHead(in_channels=emb_dim)

    def forward(self, history_BEV, BEV_queries, images, has_prev, R_curr2prev, t_curr2prev, K , R, t):

        B, n ,c, h, w = images.shape

        img_features = self.backbone(images.view(B*n, c, h, w ))

        _, C_out, H_out, W_out = img_features.shape
        images_features = img_features.reshape(B, n , C_out, H_out, W_out)

        enc_out = self.encoder(history_BEV, BEV_queries, images_features, has_prev, R_curr2prev, t_curr2prev, K, R, t)
        x = self.decoder(enc_out)
        return enc_out, x



