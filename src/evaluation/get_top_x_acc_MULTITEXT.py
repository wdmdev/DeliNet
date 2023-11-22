import torch
from typing import List
import tqdm

@torch.no_grad()
def get_top_x_acc(logits:torch.tensor,
                  top_x_percent:List[float],
                  test_loader=None,
                  vision_model=None,
                  text_model = None,
                  d = "cuda",
                  test_training_loop:bool=False) -> List[float]:
    vision_model.eval()
    text_model.eval()
    if test_loader is not None:
        img_embs = []
        text_embs = []
        text_embs2 = []
        for i, (image, text) in enumerate(tqdm.tqdm(test_loader)):
            img_emb = vision_model(image.to(d))
            text_emb = text_model(text)
            img_embs.append(img_emb)
            if isinstance(text_emb, list):
                text_embs.append(text_emb[0])
                text_embs2.append(text_emb[1])
            else:
                text_embs.append(text_emb)

            if test_training_loop and i == 2:
                break

        img_embs = torch.cat(img_embs, dim=0)
        text_embs = torch.cat(text_embs, dim=0)
        if len(text_embs2)>0:
            text_embs2 = torch.cat(text_embs2, dim=0)
            text_embs2 = text_embs2 / torch.linalg.norm(text_embs2, axis=1, keepdim=True)

        img_embs = img_embs / torch.linalg.norm(img_embs, axis=1, keepdim=True)
        text_embs = text_embs / torch.linalg.norm(text_embs, axis=1, keepdim=True)

        logits = (text_embs @ img_embs.T) * text_model.t
        if len(text_embs2)>0: logits+= (text_embs2 @ img_embs.T) * text_model.t



    n = logits.shape[0]
    accs = []
    labels = torch.arange(n).to(d)
    logtis_arg_sort = torch.argsort(logits, dim = 1, descending=True)
    for percent_acc in top_x_percent:
        if percent_acc == -1.0 or percent_acc == 0.0:
            acc = (logtis_arg_sort[:,0] == labels).to(float).mean().item()
        else:
            top_x_cols = torch.round(torch.tensor([n*percent_acc]).to(torch.float32)).to(int)
            acc = logtis_arg_sort[:, :top_x_cols] == labels[:,None]
            acc = acc.sum(dim=1).to(float).mean().item()

        accs.append(acc)

    return accs


if __name__ == "__main__":
    test = torch.rand(size=(100,100))
    print(get_top_x_acc(test, [0.0, 0.5, 0.7, 0.2]))

