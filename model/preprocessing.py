from torchvision import transforms as T


def preprocess(image):
    resize = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    resized_image = resize(image).unsqueeze(0)
    return resized_image
