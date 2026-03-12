import torch
import cv2
import numpy as np
import matplotlib
from PIL import Image
from transformers import Sam3Processor, Sam3Model

# --- Config ---
VIDEO_INPUT = "Driving Downtown - New York_480p_1.mp4"
VIDEO_OUTPUT = "output.mp4"
BATCH_SIZE = 4
TEXT_PROMPTS = ["face", "license plate"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

# --- Load model ---
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
model.eval()

# --- Open video ---
cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_INPUT}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# total_frames = 16
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {width}x{height}, {fps:.1f} fps, {total_frames} frames")
print(f"Batch size: {BATCH_SIZE}, Prompts: {TEXT_PROMPTS}")


def overlay_masks(image, masks):
    """Overlay masks onto a PIL image using matplotlib colormap."""
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]
    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

def overlay_masks_prompt(image, masks_tuple):
    """
    Overlay masks onto a PIL image using matplotlib colormap.
    masks_tuple: a list: [(frame_mask, frame_prompt_idx), ..] containing
    a tuple for each frame
    Masks belonging to the same prompt will have same color
    """
    image = image.convert("RGBA")
    # n_masks = masks.shape[0]
    n_prompts = len(TEXT_PROMPTS)
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_prompts)
    # as many colors as there are prompts
    prompt_colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_prompts)
    ]
    # masks_tuple <- [(frame_mask, frame_prompt_idx), ...]
    for masks, prompt_idx in masks_tuple:
        masks_np = 255 * masks.cpu().numpy().astype(np.uint8)
        color = prompt_colors[prompt_idx] # choose color from corresponding prompt
        for mask in masks_np:
            mask = Image.fromarray(mask)
            # color is a tuple (r,g,b), we do + (0,) to concatenate an element
            # to the tuple. The comma is for making (0,) a tuple
            overlay = Image.new("RGBA", image.size, color + (0,))
            # v: the value in the mask image, where values can either
            # be 0 or 255 (255 if we are masking that point)
            # alpha = mask.point(lambda v: int(v * 0.5))
            lut = [int(i * 0.5) for i in range(256)] # make all mask pixels semi-transparent
            alpha = mask.point(lut)
            # set the mask in the alpha channel
            overlay.putalpha(alpha)
            # blend overlay (mask) into the image
            image = Image.alpha_composite(image, overlay)
        
    return image


def process_batch(batch_pil, prompt):
    """Run SAM3 on a batch of frames with a single text prompt."""
    text_list = [prompt] * len(batch_pil)

    inputs = processor(
        images=batch_pil,
        text=text_list,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        # threshold=THRESHOLD,
        # mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist(),
    )

    return [r["masks"] for r in results]


# --- Per-prompt pass, streaming frames, store only masks ---
# masks_per_frame[i] collects mask tensors (on CPU) across prompt passes
masks_per_frame = [[] for _ in range(total_frames)]

for prompt_idx, prompt in enumerate(TEXT_PROMPTS):
    print(f"\nPass {prompt_idx + 1}/{len(TEXT_PROMPTS)}: prompt='{prompt}'")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0 # to count frame index when going to next batch
    batch_pil = []
    cnt = 0
    while True:
        # cnt+=1
        # if cnt > total_frames:
        #     break
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        batch_pil.append(Image.fromarray(frame_rgb))

        if len(batch_pil) == BATCH_SIZE:
            batch_masks = process_batch(batch_pil, prompt)
            for i, masks in enumerate(batch_masks):
                if len(masks) > 0:
                    # masks_per_frame[frame_idx + i].append(masks.cpu())
                    # save prompt_idx for the (per prompt) segmentation color
                    masks_per_frame[frame_idx + i].append((masks.cpu(), prompt_idx))
            frame_idx += len(batch_pil)
            batch_pil.clear()
            torch.cuda.empty_cache()
            print(f"  Processed {frame_idx}/{total_frames} frames", end="\r")

    # Remainder
    if batch_pil:
        batch_masks = process_batch(batch_pil, prompt)
        for i, masks in enumerate(batch_masks):
            if len(masks) > 0:
                masks_per_frame[frame_idx + i].append(masks.cpu())
        frame_idx += len(batch_pil)
        torch.cuda.empty_cache()

    print(f"  Processed {frame_idx}/{total_frames} frames")

# --- Final pass: re-read frames, merge masks, write output ---
print("\nWriting output...")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(total_frames):
    ret, frame_bgr = cap.read()
    if not ret:
        break

    # if mask exists (entry is not empty)
    if masks_per_frame[i]:
        # all_masks = torch.cat(masks_per_frame[i], dim=0)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        # overlaid = overlay_masks(frame_pil, all_masks)
        # Remember: masks_per_frame[i] is a single entry containing
        # (mask, prompt_idx) for the i-th frame
        overlaid = overlay_masks_prompt(frame_pil, masks_per_frame[i])
        bgr = cv2.cvtColor(np.array(overlaid.convert("RGB")), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    else:
        writer.write(frame_bgr)

cap.release()
writer.release()
print(f"Done. Processed {total_frames} frames, {len(TEXT_PROMPTS)} prompts. Output: {VIDEO_OUTPUT}")