import torch
import torch.nn.functional as F

def get_video_masks_features(
    frames,
    T_slice,
    seg_model,
    seg_preprocess,
    person_idx: int,
    seg_batch: int = 16,
    mask_latent_size: tuple[int, int] = (64, 64),
    device: str = "cpu",
):
    # Convert to torch [T,3,H,W]
    imgs = torch.from_numpy(frames).permute(0, 3, 1, 2)  # uint8
    # Apply preprocess per frame (keeps aspect ratio; short side ~520)
    inp_list = [seg_preprocess(img) for img in imgs]     # list of [3,H',W']
    inp = torch.stack(inp_list, dim=0).to(device)        # [T,3,H',W']

    all_masks = []
    for s in range(0, T_slice, seg_batch):
        chunk = inp[s : s + seg_batch]                   # [b,3,H',W']
        out = seg_model(chunk)["out"]                    # [b,C,H',W']
        probs = out.softmax(dim=1)                       # [b,C,H',W']
        person = probs[:, person_idx:person_idx+1]       # [b,1,H',W']

        # Downsample to latent 64x64 with nearest
        person_64 = F.interpolate(
            person,
            size=mask_latent_size,
            mode="nearest",
        ).squeeze(1)                                     # [b,64,64]

        all_masks.append(person_64)

    masks = torch.cat(all_masks, dim=0)
    return masks
