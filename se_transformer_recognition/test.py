from recognition import recognize

# Basic usage (uses defaults defined in the file)
text = recognize("test_img_1.png")
print(text)


# images = ["img1.png", "img2.png", "img3.png"]
# for img in images:
#     print(f"{img}: {recognize(img)}")


# text_custom = recognize("test_image.png", beam_width=5, model_path="other_model.pth")