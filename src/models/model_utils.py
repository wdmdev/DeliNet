import torch
import copy

class ResNet_wrapper(torch.nn.Module):
    def __init__(self, ResNet, latent_dim=768, d="cuda"):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.model = ResNet
        self.activation = torch.nn.ReLU()

        if self.model.name_or_path.endswith(str(50)):
            self.fc = torch.nn.Linear(2048, latent_dim)

        elif self.model.name_or_path.endswith(str(18)):
            self.fc = torch.nn.Linear(512, latent_dim)

    def forward(self, images):
        output = self.model(images)
        output = output.pooler_output.squeeze()
        out = self.fc(self.activation(output))

        return out

class ViT_wrapper(torch.nn.Module):
    def __init__(self, ViT, latent_dim=768, d="cuda"):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.model = ViT
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(latent_dim, latent_dim)

    def forward(self, images):
        output = self.model(images)
        #latent = output.latent
        latent = output.pooler_output
        out = self.fc(self.activation(latent))

        return out


class Bert_mono_wrapper(torch.nn.Module):
    def __init__(self, BertModel, BertTokenizer, latent_dim=768, d="cuda", max_length=16):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel = BertModel
        self.BertTokenizer = BertTokenizer
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, titles, ingredients=None):
        preprocessed = self.BertTokenizer(text=titles,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        cls_token = output.pooler_output
        out = self.fc(self.activation(cls_token))

        return out

class Bert_2xinput_wrapper(torch.nn.Module):
    def __init__(self, BertModel, BertTokenizer, latent_dim=768, d="cuda", max_length=128):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel = BertModel
        self.BertTokenizer = BertTokenizer
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, titles, ingres, desc):
        combined = [title + " " + ing for title, ing in zip(titles, ingres)]
        preprocessed = self.BertTokenizer(text=combined,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        #output = self.BertModel(preprocessed.input_ids)
        cls_token = output.pooler_output
        out = self.fc(self.activation(cls_token))

        return out

class Bert_2x_network_wrapper(torch.nn.Module):
    def __init__(self, BertModel, BertTokenizer, latent_dim=768, d="cuda", max_length=32):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel1 = BertModel
        self.BertModel2 = copy.deepcopy(BertModel)

        self.BertTokenizer = BertTokenizer
        self.activation = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(latent_dim*2, latent_dim*2)
        self.fc2 = torch.nn.Linear(latent_dim*2, latent_dim)

        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, titles, ingredients):
        titles = self.BertTokenizer(text=titles,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length,
                                    return_tensors="pt").to(self.d)

        ingredients = self.BertTokenizer(text=ingredients,
                                         padding=True,
                                         truncation=True,
                                         max_length=self.max_length*4,
                                         return_tensors="pt").to(self.d)
        #preprocessed["input_ids"]
        output1 = self.BertModel1(**titles).pooler_output
        output2 = self.BertModel2(**ingredients).pooler_output

        output = self.fc1(self.activation(torch.cat((output1, output2), dim=1)))
        out = self.fc2(self.activation(output))

        return out