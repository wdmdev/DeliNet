import torch
import copy
from transformers import (DistilBertTokenizerFast, DistilBertModel,
                          ResNetModel, ViTModel, EfficientNetModel,
                          EfficientFormerForImageClassification,
                          BertModel, BertTokenizerFast, AutoModelForImageClassification,
                          ConvNextV2Model, ConvNextForImageClassification, CLIPModel, CLIPProcessor)

from peft import LoraConfig, get_peft_model

class CLIP_vision_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu"):
        super().__init__()
        self.size = ""
        self.d = d
        self.latent_dim = latent_dim
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.out_proj = self.model.visual_projection
        self.model = self.model.vision_model

    def forward(self, images):
        latent = self.model(images).pooler_output
        latent = self.out_proj(latent)
        return latent

class CLIP_text_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu"):
        super().__init__()
        self.size = ""
        self.d = d
        self.latent_dim = latent_dim
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.out_proj = self.model.text_projection
        self.t = self.model.logit_scale

        self.model = self.model.text_model
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, text):
        titles, ingre, desc = text
        titles_emb = self.processor(text=titles, return_tensors="pt", padding=True)
        latent = self.model(**titles_emb).pooler_output
        latent = self.out_proj(latent)
        return latent

class EfficientTrans_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu"):
        super().__init__()
        self.size = ""
        self.d = d
        self.latent_dim = latent_dim
        self.model = EfficientFormerForImageClassification.from_pretrained("snap-research/efficientformer-l1-300")
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(1000, latent_dim)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, images):
        latent = self.model(images)
        #latent = output.latent
        latent = latent.logits
        latent = self.dropout(latent)
        latent = self.activation(latent)
        latent = self.fc(latent)


        return latent

class Swin_Transformer_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu"):
        super().__init__()
        self.size = ""
        self.d = d
        self.latent_dim = latent_dim
        self.model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(1000, latent_dim)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, images):
        latent = self.model(images)
        #latent = output.latent
        latent = latent.logits
        latent = self.dropout(latent)
        latent = self.activation(latent)
        latent = self.fc(latent)

        return latent

class ConvNextTiny_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", classification=False):
        super().__init__()
        self.size = ""
        self.classification = classification
        self.d = d
        self.latent_dim = latent_dim
        #self.model = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224")
        self.model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

        self.id2label = self.model.config.id2label
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(1000, latent_dim)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, images):
        latent = self.model(images)
        #latent = output.latent
        latent = latent.logits
        if self.classification:
            return latent

        latent = self.dropout(latent)
        latent = self.activation(latent)
        latent = self.fc(latent)

        return latent

class ConvNext_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", classification=False):
        super().__init__()
        self.size = ""
        self.classification = classification
        self.d = d
        self.latent_dim = latent_dim
        #self.model = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224")
        self.model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224-22k")

        self.id2label = self.model.config.id2label
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(21_841, latent_dim)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, images):
        latent = self.model(images)
        #latent = output.latent
        latent = latent.logits
        if self.classification:
            return latent

        latent = self.dropout(latent)
        latent = self.activation(latent)
        latent = self.fc(latent)

        return latent


class EfficientTransNonPre_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu"):
        super().__init__()
        self.size = ""
        self.d = d
        self.latent_dim = latent_dim
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(1000, latent_dim)
        self.dropout = torch.nn.Dropout(0.1)

        from transformers import EfficientFormerConfig
        print("using non pretrained EfficientTrans")
        config = EfficientFormerConfig(num_labels=1000)
        self.model = EfficientFormerForImageClassification(config)

    def forward(self, images):
        latent = self.model(images)
        #latent = output.latent
        latent = latent.logits
        latent = self.dropout(latent)
        latent = self.activation(latent)
        latent = self.fc(latent)

        return latent

class Efficientnet_wrapper(torch.nn.Module):
    def __init__(self, size = "4", latent_dim=768, d="cpu"):
        super().__init__()
        self.size = str(size)
        self.d = d
        self.latent_dim = latent_dim
        self.model = EfficientNetModel.from_pretrained(f"google/efficientnet-b{self.size}")
        self.activation = torch.nn.ReLU()
        if self.size == "3":
            self.fc = torch.nn.Linear(1536, latent_dim)
        if self.size == "4":
            self.fc = torch.nn.Linear(1792, latent_dim)
        if self.size == "5":
            self.fc = torch.nn.Linear(2048, latent_dim)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, images):
        latent = self.model(images)
        #latent = output.latent
        latent = latent.pooler_output
        latent = self.dropout(latent)
        latent = self.activation(latent)
        latent = self.fc(latent)

        return latent


class ResNet50NonPre_wrapper(torch.nn.Module):
    def __init__(self, size="50", latent_dim=768, d="cpu", pretrained=True):
        super().__init__()
        self.size = str(size)
        self.d = d
        self.latent_dim = latent_dim
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(2048, latent_dim)

        assert self.size == "50"
        from transformers import ResNetConfig
        print("using non pretrained resnet50")
        resnet_conf = ResNetConfig()
        self.model = ResNetModel(resnet_conf)



    def forward(self, images):
        output = self.model(images)
        output = output.pooler_output.squeeze()
        output = self.dropout(output)
        output = self.activation(output)
        output = self.fc(output)

        return output

class ResNet_wrapper(torch.nn.Module):
    def __init__(self, size="50", latent_dim=768, d="cpu", pretrained=True):
        super().__init__()
        self.size = str(size)
        self.d = d
        self.latent_dim = latent_dim
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)

        if self.size == "50":
            self.fc = torch.nn.Linear(2048, latent_dim)

        elif self.size == "18":
            self.fc = torch.nn.Linear(512, latent_dim)

        else:
            raise AssertionError(f"pick a proper resnet size and not {self.size}")
        self.model = ResNetModel.from_pretrained(f"microsoft/resnet-{self.size}")



    def forward(self, images):
        output = self.model(images)
        output = output.pooler_output.squeeze()
        output = self.dropout(output)
        output = self.activation(output)
        output = self.fc(output)

        return output

class ViT_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu"):
        super().__init__()
        self.size = ""
        self.d = d
        self.latent_dim = latent_dim
        #self.model = ViTModel.from_pretrained("facebook/vit-mae-base")

        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, images):
        latent = self.model(images)
        #latent = output.latent
        latent = latent[0][:,0,:]
        latent = self.dropout(latent)
        latent = self.activation(latent)
        latent = self.fc(latent)

        return latent


class Bert_mono_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=32):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel = BertModel.from_pretrained("bert-base-uncased")
        self.BertTokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text
        preprocessed = self.BertTokenizer(text=titles,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        output = output.pooler_output
        output = self.dropout(output)
        output = self.fc(self.activation(output))

        return output

class DistilBert_mono_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=32):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07]))  # init value from clip paper
        self.BertModel = DistilBertModel.from_pretrained("distilbert-base-uncased")


    def forward(self, text):
        titles, _, _ = text
        preprocessed = self.BertTokenizer(text=titles,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        output = output[0][:,0,:]
        output = self.dropout(output)
        output = self.fc(self.activation(output))

        return output

class DistilBert_monoTextAug_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=32):
        super().__init__()
        from niacin.augment import RandAugment
        from niacin.text import en
        self.aug = RandAugment([en.add_hyponyms, en.add_hypernyms, en.add_synonyms], n=1, m=100)
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07]))  # init value from clip paper
        self.BertModel = DistilBertModel.from_pretrained("distilbert-base-uncased")


    def forward(self, text):
        titles, _, _ = text
        print(titles[:3])
        if self.training:
            titles = [next(iter(self.aug))(title) for title in titles]
        print(titles[:3])

        preprocessed = self.BertTokenizer(text=titles,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        output = output[0][:,0,:]
        output = self.dropout(output)
        output = self.fc(self.activation(output))

        return output

class DistilBert_mono_NonPre_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=32, pretrained=True):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07]))  # init value from clip paper

        print("using non pretrained DistillBert")
        from transformers import DistilBertConfig
        configuration = DistilBertConfig()
        self.BertModel = DistilBertModel(configuration)


    def forward(self, text):
        titles, _, _ = text
        preprocessed = self.BertTokenizer(text=titles,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        output = output[0][:,0,:]
        output = self.dropout(output)
        output = self.fc(self.activation(output))

        return output



class DistilBertLora_mono_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=32):
        super().__init__()

        lora_config = LoraConfig(r=32, lora_alpha=1, lora_dropout=0.1,
                                 target_modules=["q_lin", "k_lin", "v_lin", "out_lin"])
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.BertModel = get_peft_model(self.BertModel, lora_config)
        print("running LORA", self.BertModel.print_trainable_parameters())
        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07]))  # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text
        preprocessed = self.BertTokenizer(text=titles,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        output = output[0][:,0,:]
        output = self.dropout(output)
        output = self.fc(self.activation(output))

        return output

class Bert_2xInp_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=128):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel = BertModel.from_pretrained("bert-base-uncased")
        self.BertTokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text

        combined = [title + " " + ing for title, ing in zip(titles, ingres)]
        preprocessed = self.BertTokenizerFast(text=combined,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        #output = self.BertModel(preprocessed.input_ids)output[0][:,0,:]
        cls_token = output.pooler_output
        out = self.fc(self.activation(cls_token))

        return out





class Bert_2xNet_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=32):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel1 = BertModel.from_pretrained("bert-base-uncased")
        self.BertModel2 = copy.deepcopy(BertModel)

        self.BertTokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.activation = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(latent_dim*2, latent_dim*2)
        self.fc2 = torch.nn.Linear(latent_dim*2, latent_dim)

        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text
        titles = self.BertTokenizer(text=titles,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length,
                                    return_tensors="pt").to(self.d)

        ingredients = self.BertTokenizer(text=ingres,
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

class DistilBert_3xNet_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=[32, 256, 256]):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.BertModel2 = copy.deepcopy(self.BertModel1)
        self.BertModel3 = copy.deepcopy(self.BertModel1)

        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = torch.nn.Linear(latent_dim*3, latent_dim*3)
        self.fc2 = torch.nn.Linear(latent_dim*3, latent_dim)

        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text
        titles = self.BertTokenizer(text=titles,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length[0],
                                    return_tensors="pt").to(self.d)

        ingres = self.BertTokenizer(text=ingres,
                                         padding=True,
                                         truncation=True,
                                         max_length=self.max_length[1],
                                         return_tensors="pt").to(self.d)

        descr = self.BertTokenizer(text=descr,
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length[2],
                                   return_tensors="pt").to(self.d)

        #preprocessed["input_ids"]
        output1 = self.BertModel1(**titles)[0][:,0,:] # take all batches, first row, all columns
        output2 = self.BertModel2(**ingres)[0][:,0,:] # take all batches, first row, all columns
        output3 = self.BertModel3(**descr)[0][:,0,:] # take all batches, first row, all columns

        output = torch.cat((output1, output2, output3), dim=1)
        output = self.dropout(self.activation(output))
        output = self.fc1(output)
        output = self.dropout(self.activation(output))
        output = self.fc2(output)

        return output

class DistilBert_3xNet3xOutWmix_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=[32, 128, 128], w = [1.0, 0.66, 0.33]):
        super().__init__()
        self.w = w
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.BertModel2 = copy.deepcopy(self.BertModel1)
        self.BertModel3 = copy.deepcopy(self.BertModel1)

        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = torch.nn.Linear(latent_dim, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, latent_dim)
        self.fc3 = torch.nn.Linear(latent_dim, latent_dim)
        #self.fc2 = torch.nn.Linear(latent_dim*3, latent_dim)

        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text
        titles = self.BertTokenizer(text=titles,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length[0],
                                    return_tensors="pt").to(self.d)

        ingres = self.BertTokenizer(text=ingres,
                                         padding=True,
                                         truncation=True,
                                         max_length=self.max_length[1],
                                         return_tensors="pt").to(self.d)

        descr = self.BertTokenizer(text=descr,
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length[2],
                                   return_tensors="pt").to(self.d)

        #preprocessed["input_ids"]
        output1 = self.BertModel1(**titles)[0][:,0,:] # take all batches, first row, all columns
        output2 = self.BertModel2(**ingres)[0][:,0,:] # take all batches, first row, all columns
        output3 = self.BertModel3(**descr)[0][:,0,:] # take all batches, first row, all columns

        output1 = self.fc1(self.activation(self.dropout(output1)))
        output2 = self.fc2(self.activation(self.dropout(output2)))
        output3 = self.fc3(self.activation(self.dropout(output3)))

        if self.training:
            return [output1, output2, output3]
        else:
            return ((output1 / torch.linalg.norm(output1, axis=1, keepdim=True))*self.w[0] +
                    (output2 / torch.linalg.norm(output2, axis=1, keepdim=True))*self.w[1] +
                    (output3 / torch.linalg.norm(output3, axis=1, keepdim=True))*self.w[2]
                    )

class DistilBert_3xNet3xOutWCons_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=[32, 128, 128], w = [1.0, 1.0, 1.0]):
        super().__init__()
        self.w = w
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.BertModel2 = copy.deepcopy(self.BertModel1)
        self.BertModel3 = copy.deepcopy(self.BertModel1)

        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = torch.nn.Linear(latent_dim, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, latent_dim)
        self.fc3 = torch.nn.Linear(latent_dim, latent_dim)
        #self.fc2 = torch.nn.Linear(latent_dim*3, latent_dim)

        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text
        titles = self.BertTokenizer(text=titles,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length[0],
                                    return_tensors="pt").to(self.d)

        ingres = self.BertTokenizer(text=ingres,
                                         padding=True,
                                         truncation=True,
                                         max_length=self.max_length[1],
                                         return_tensors="pt").to(self.d)

        descr = self.BertTokenizer(text=descr,
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length[2],
                                   return_tensors="pt").to(self.d)

        #preprocessed["input_ids"]
        output1 = self.BertModel1(**titles)[0][:,0,:] # take all batches, first row, all columns
        output2 = self.BertModel2(**ingres)[0][:,0,:] # take all batches, first row, all columns
        output3 = self.BertModel3(**descr)[0][:,0,:] # take all batches, first row, all columns

        output1 = self.fc1(self.activation(self.dropout(output1)))
        output2 = self.fc2(self.activation(self.dropout(output2)))
        output3 = self.fc3(self.activation(self.dropout(output3)))

        if self.training:
            return [output1, output2, output3]
        else:
            return ((output1 / torch.linalg.norm(output1, axis=1, keepdim=True))*self.w[0] +
                    (output2 / torch.linalg.norm(output2, axis=1, keepdim=True))*self.w[1] +
                    (output3 / torch.linalg.norm(output3, axis=1, keepdim=True))*self.w[2]
                    )

class DistilBert_3xInp_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=256):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.activation = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(latent_dim, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text

        combined = [title + " " + ing + " " + des for title, ing, des in zip(titles, ingres, descr)]
        preprocessed = self.BertTokenizer(text=combined,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)[0][:,0,:]
        output = self.dropout(self.activation(output))
        output = self.fc1(output)
        output = self.dropout(self.activation(output))
        output = self.fc2(output)

        return output

class DistilBert_2xInp_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=128):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.activation = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(latent_dim, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text

        combined = [title + " " + ing for title, ing in zip(titles, ingres)]
        preprocessed = self.BertTokenizer(text=combined,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)[0][:,0,:]
        output = self.dropout(self.activation(output))
        output = self.fc1(output)
        output = self.dropout(self.activation(output))
        output = self.fc2(output)

        return output

class DistilBert_2xNet_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=[32, 256]):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.BertModel2 = copy.deepcopy(self.BertModel1)

        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = torch.nn.Linear(latent_dim*2, latent_dim*2)
        self.fc2 = torch.nn.Linear(latent_dim*2, latent_dim)

        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text
        titles = self.BertTokenizer(text=titles,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length[0],
                                    return_tensors="pt").to(self.d)

        ingres = self.BertTokenizer(text=ingres,
                                         padding=True,
                                         truncation=True,
                                         max_length=self.max_length[1],
                                         return_tensors="pt").to(self.d)


        #preprocessed["input_ids"]
        output1 = self.BertModel1(**titles)[0][:,0,:] # take all batches, first row, all columns
        output2 = self.BertModel2(**ingres)[0][:,0,:] # take all batches, first row, all columns

        output = torch.cat((output1, output2), dim=1)
        output = self.dropout(self.activation(output))
        output = self.fc1(output)
        output = self.dropout(self.activation(output))
        output = self.fc2(output)

        return output

class DistilBert_2xNet_test_wrapper(torch.nn.Module):
    def __init__(self, latent_dim=768, d="cpu", max_length=[32, 256]):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.BertModel2 = copy.deepcopy(self.BertModel1)

        self.BertTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        # self.activation = torch.nn.GELU()
        # self.dropout = torch.nn.Dropout(0.1)
        # self.fc1 = torch.nn.Linear(latent_dim*2, latent_dim*2)
        # self.fc2 = torch.nn.Linear(latent_dim*2, latent_dim)

        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, text):
        titles, ingres, descr = text
        titles = self.BertTokenizer(text=titles,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length[0],
                                    return_tensors="pt").to(self.d)

        ingres = self.BertTokenizer(text=ingres,
                                         padding=True,
                                         truncation=True,
                                         max_length=self.max_length[1],
                                         return_tensors="pt").to(self.d)


        #preprocessed["input_ids"]
        output1 = self.BertModel1(**titles)[0][:,0,:] # take all batches, first row, all columns
        output2 = self.BertModel2(**ingres)[0][:,0,:] # take all batches, first row, all columns

        # output = torch.cat((output1, output2), dim=1)
        # output = self.dropout(self.activation(output))
        # output = self.fc1(output)
        # output = self.dropout(self.activation(output))
        # output = self.fc2(output)

        return [output1, output2]