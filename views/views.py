from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt

def visualize_predictions(fold, image_path, predictions, ground_truth_target=None, score_threshold=0.5):
    # Load the original image for visualization
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Plot Ground Truth Boxes if provided
    if ground_truth_target is not None:
        gt_boxes = ground_truth_target['boxes'].cpu().numpy()
        for i in range(len(gt_boxes)):
            gx1, gy1, gx2, gy2 = gt_boxes[i]
            draw.rectangle([(gx1, gy1), (gx2, gy2)], outline="green", width=3)
            # Optionally add text for ground truth
            try:
                font = ImageFont.truetype("arial.ttf", 20) # You might need to specify a path to a font file
            except IOError:
                font = ImageFont.load_default()
            text_x_gt = gx1
            text_y_gt = gy1 - 20 if gy1 - 20 > 0 else gy1 + 5
            draw.rectangle([(text_x_gt, text_y_gt), (text_x_gt + 100, text_y_gt + 20)], fill="green")
            draw.text((text_x_gt, text_y_gt), "Ground Truth", fill="white", font=font)

    # Plot Predicted Boxes
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    # Assuming label 1 means 'cow'
    class_names = {1: 'cow'}

    for i in range(len(boxes)):
        if scores[i] > score_threshold:
            x1, y1, x2, y2 = boxes[i]
            label = labels[i]
            score = scores[i]

            # Draw bounding box
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

            # Prepare text for label and score
            display_text = f"{class_names.get(label, 'Unknown')}: {score:.2f}"

            # Determine text position
            text_x = x1
            text_y = y1 - 20 if y1 - 20 > 0 else y1 + 5

            # Draw text background (optional, for better readability)
            try:
                font = ImageFont.truetype("arial.ttf", 20) # You might need to specify a path to a font file
            except IOError:
                font = ImageFont.load_default()

            text_width, text_height = draw.textbbox((0,0), display_text, font=font)[2:]
            draw.rectangle([(text_x, text_y), (text_x + text_width, text_y + text_height)], fill="red")
            draw.text((text_x, text_y), display_text, fill="white", font=font)

    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Predicted (Red) vs. Ground Truth (Green) Bounding Boxes')
    plt.savefig(f"outputs\\ssd_mobilenet\\resultado_fold_{fold}.png")
