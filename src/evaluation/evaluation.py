# For this project, a function will be made for evaluation.
# This function will compute a metric for comparing the self-trained model and CLIP.
# The metric is the mean cosine similarity between the image and recipe embeddings.
# The cosine similarity is computed as the dot product of the embeddings divided by the product of the norms of the embeddings.
# The cosine similarity is a measure of similarity between two non-zero vectors.


import unittest
import numpy as np
import torch




# Define the evaluation function
def evaluate(model, dataloader, device, similarity='cosine', acc_percent=0.01):

    ## First we compute all the embeddings for the images and recipes
    model.eval()

    # Compute the embeddings for the images
    image_embeddings = []
    for batch in dataloader:
        image, _, _ = batch
        image = image.to(device)
        image_emb = model.image_encoder(image)
        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
        image_embeddings.append(image_emb)

    # Compute the embeddings for the recipes
    recipe_embeddings = []
    for batch in dataloader:
        _, input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        recipe_emb = model.text_encoder(input_ids, attention_mask)
        recipe_emb = recipe_emb / recipe_emb.norm(dim=1, keepdim=True)
        recipe_embeddings.append(recipe_emb)

    # Compute the mean cosine similarity between the image and recipe embeddings
    cosine_similarity = 0
    for image_emb, recipe_emb in zip(image_embeddings, recipe_embeddings):
        if similarity == 'cosine':
            cosine_similarity += (image_emb * recipe_emb).sum(dim=1).mean().item()
        elif similarity == 'euclidean':
            cosine_similarity += (image_emb - recipe_emb).pow(2).sum(1).mean().item()
    

    # Compute the accuracy of the model based on top acc_percent percentage of the most similar images
    cosine_similarity /= len(dataloader)
    if similarity == 'cosine':
        similarity = True
    elif similarity == 'euclidean':
        similarity = False
    topk = int(len(dataloader) * acc_percent)
    accuracy = 0
    for image_emb, recipe_emb in zip(image_embeddings, recipe_embeddings):
        similarity_scores = []
        for recipe in recipe_embeddings:
            if similarity:
                similarity_scores.append((image_emb * recipe).sum(dim=1).item())
            else:
                similarity_scores.append((image_emb - recipe).pow(2).sum(1).item())
        similarity_scores = torch.tensor(similarity_scores)
        _, topk_indices = similarity_scores.topk(topk)
        if 0 in topk_indices:
            accuracy += 1
    accuracy /= len(dataloader)

    return cosine_similarity, accuracy



def evaluate_model(image_embeddings, recipe_embeddings):
    total_cosine_similarity = 0
    num_pairs = len(image_embeddings)

    for img_emb, rec_emb in zip(image_embeddings, recipe_embeddings):
        cosine_similarity = np.dot(img_emb, rec_emb) / (np.linalg.norm(img_emb) * np.linalg.norm(rec_emb))
        total_cosine_similarity += cosine_similarity

    mean_cosine_similarity = total_cosine_similarity / num_pairs
    return mean_cosine_similarity


# class TestEvaluateModel(unittest.TestCase):
#     def test_evaluate_model(self):
#         image_embeddings = [np.array([1, 2, 3]), np.array([4, 5, 6])]
#         recipe_embeddings = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        
#         result = evaluate_model(image_embeddings, recipe_embeddings)
#         self.assertEqual(result, 1.0)


class MockModel:
    def __init__(self):
        self.device = torch.device('cpu')

    def eval(self):
        pass

    def image_encoder(self, image):
        return torch.ones((1, 128), device=self.device)

    def text_encoder(self, input_ids, attention_mask):
        return torch.ones((1, 128), device=self.device)

class MockDataLoader:
    def __init__(self, num_batches):
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield torch.ones((1, 3, 224, 224)), torch.ones((1, 100)), torch.ones((1, 100))

class TestEvaluate(unittest.TestCase):
    def test_evaluate(self):
        model = MockModel()
        dataloader = MockDataLoader(10)
        device = torch.device('cpu')

        cosine_similarity, accuracy = evaluate(model, dataloader, device)
        
        self.assertIsInstance(cosine_similarity, float)
        self.assertIsInstance(accuracy, float)

        self.assertTrue(0 <= cosine_similarity <= 1)
        self.assertTrue(0 <= accuracy <= 1)


if __name__ == '__main__':
    unittest.main()