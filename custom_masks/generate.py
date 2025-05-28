import cv2
import numpy as np
from pathlib import Path
import argparse

def draw_mask(image_path, cursor_size):
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Resize the image to 5x its original size
    height, width = image.shape[:2]
    image = cv2.resize(image, (width * 5, height * 5), interpolation=cv2.INTER_NEAREST)

    # Create a window and set mouse callback
    cv2.namedWindow("Draw Mask")
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    drawing = False
    overlay = image.copy()

    def draw(event, x, y, flags, param):
        nonlocal drawing, overlay
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(mask, (x, y), cursor_size, (255), -1)
                # Update only the changed region
                roi = overlay[max(0, y-cursor_size):min(overlay.shape[0], y+cursor_size+1),
                              max(0, x-cursor_size):min(overlay.shape[1], x+cursor_size+1)]
                roi[mask[max(0, y-cursor_size):min(mask.shape[0], y+cursor_size+1),
                         max(0, x-cursor_size):min(mask.shape[1], x+cursor_size+1)] == 255] = [0, 0, 255]
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        
        # Show the mask overlaid on the image
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        cv2.imshow("Draw Mask", image)

    cv2.setMouseCallback("Draw Mask", draw)

    print(f"Draw the mask using the left mouse button. Cursor size: {cursor_size}")
    print("Press 's' to save, 'r' to reset, '+' to increase cursor size, '-' to decrease cursor size, or 'q' to quit.")

    while True:
        cv2.imshow("Draw Mask", image)
        key = cv2.waitKey(3_000) & 0xFF

        if key == ord('s'):
            # Save the mask (resize back to original size)
            mask_small = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
            cv2.imwrite(str(mask_path), mask_small)
            print(f"Mask saved as {mask_path}")
            break
        elif key == ord('r'):
            # Reset the mask
            mask.fill(0)
            image = cv2.imread(str(image_path))
            image = cv2.resize(image, (width * 5, height * 5), interpolation=cv2.INTER_NEAREST)
            overlay = image.copy()
        elif key == ord('+'):
            cursor_size += 1
            print(f"Cursor size increased to {cursor_size}")
        elif key == ord('-'):
            cursor_size = max(1, cursor_size - 1)
            print(f"Cursor size decreased to {cursor_size}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Draw binary mask on an image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--cursor_size", type=int, default=40, help="Initial cursor size (default: 5)")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return

    draw_mask(image_path, args.cursor_size)

if __name__ == "__main__":
    main()
