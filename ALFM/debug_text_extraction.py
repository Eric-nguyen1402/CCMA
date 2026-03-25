import torch
import torch.nn.functional as F
from dotenv import load_dotenv
import open_clip

def debug_text_generation():
    load_dotenv()
    
    # Load the exact same model used for image features
    model_name = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k" 
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # CIFAR-100 classes in exact dataset order
    cifar100_classes = [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
        "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
        "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
        "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
        "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
        "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
        "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
    ]
    
    # Generate text embeddings exactly as they should be
    templates = ["a photo of a {}".format]  # simple template
    
    text_features = []
    model.eval()
    
    for i, class_name in enumerate(cifar100_classes):
        # Clean class name (replace underscores)
        clean_name = class_name.replace("_", " ")
        text = f"a photo of a {clean_name}"
        
        tokens = tokenizer([text])
        with torch.no_grad():
            text_feat = model.encode_text(tokens)
            text_feat = F.normalize(text_feat, dim=1)
            text_features.append(text_feat)
            
        if i < 5:  # debug first few
            print(f"Class {i} ({class_name}): '{text}' -> shape {text_feat.shape}")
    
    text_embeddings = torch.cat(text_features, dim=0)  # (100, 768)
    print(f"Final text embeddings shape: {text_embeddings.shape}")
    
    # Save manually for testing
    import h5py, os
    from dotenv import dotenv_values
    env = dotenv_values()
    
    # Create a test file
    test_file = os.path.join(env["FEATURE_CACHE_DIR"], "cifar100", "test_text.hdf")
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    with h5py.File(test_file, "w") as f:
        text_group = f.create_group("text")
        text_group.create_dataset("features", data=text_embeddings.cpu().numpy())
        text_group.create_dataset("labels", data=torch.arange(100).unsqueeze(1).numpy())
        
    print(f"Saved test text embeddings to: {test_file}")

if __name__ == "__main__":
    debug_text_generation()