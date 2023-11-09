import torch
from typing import List
import tqdm

def get_top_x_acc(logits:torch.tensor,
                  top_x_percent:List[float],
                  test_loader=None,
                  vision_model=None,
                  text_model = None,
                  d = "cuda") -> List[float]:

    with torch.no_grad():
        vision_model.eval()
        text_model.eval()
        #todo this might run of out GPU memory for large testsets but atleast its fast. maybe
        if test_loader is not None:
            img_logits = []
            text_logits = []
            for image, text in tqdm.tqdm(test_loader):
                image_encoding = vision_model(image.to(d))
                text_encoding = text_model(text)
                img_logits.append(image_encoding)
                text_logits.append(text_encoding)

            img_logits = torch.cat(img_logits, dim=0)
            text_logits = torch.cat(text_logits, dim=0)

            img_logits = img_logits / torch.linalg.norm(img_logits, axis=1, keepdim=True)
            text_logits = text_logits / torch.linalg.norm(text_logits, axis=1, keepdim=True)

            logits = text_logits @ img_logits.T

        n = logits.shape[0]
        accs = []
        labels = torch.arange(n).to(d)
        logtis_arg_sort = torch.argsort(logits, dim = 1, descending=True)
        for percent_acc in top_x_percent:
            if percent_acc == -1.0 or percent_acc == 0.0:
                acc = (logtis_arg_sort[:,0] == labels).to(float).mean().item()
            else:
                top_x_cols = torch.round(torch.tensor([n*percent_acc])).to(int)
                acc = logtis_arg_sort[:, :top_x_cols] == labels[:,None]
                acc = acc.sum(dim=1).to(float).mean().item()

            accs.append(acc)

    return accs


if __name__ == "__main__":
    test = torch.rand(size=(100,100))
    print(get_top_x_acc(test, [0.0, 0.5, 0.7, 0.2]))


