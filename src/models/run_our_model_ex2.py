from src.models.our_model import train_our_model
import os
from src.models.model_utils import *



if __name__ == "__main__":
  current_dir = os.path.dirname(os.path.abspath(__file__))
  food_dir = os.path.join(current_dir, '..', '..', 'data', 'processed', 'KaggleFoodDataset')
  csv_file_path = os.path.join(food_dir, 'data.csv')
  image_dir = os.path.join(food_dir, 'images')

  d = "cuda"
  vision_model = EfficientTrans_wrapper()
  save_results = True


  text_models = [
                 (DistilBert_2xInp_wrapper(max_length=256), 50),
                 (DistilBert_3xInp_wrapper(max_length=512), 50),
                 (DistilBert_2xNet_wrapper(max_length=[32, 256]), 50),
                 (DistilBert_3xNet_wrapper(max_length=[32, 256, 256]), 50),
                 ]
  #running test to see that all models can run
  training_loop_test = True
  for text_model, batch_size in text_models:
    vision_model = EfficientTrans_wrapper()
    print("####"*20)
    print("####"*20)
    train_our_model(csv_file_path,
                    image_dir,
                    vision_model,
                    text_model,
                    batch_size=batch_size,
                    lr=0.0001,
                    d=d,
                    num_epochs=100,
                    use_mixed_precision = True,
                    max_time=3600*2,  # 1hour max
                    training_loop_test=training_loop_test,
                    save_results=save_results)
    text_model = text_model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

  print("!!!!!!!!!!!!!"*20)
  print("!!!!!!!!!!!!!"*20)
  print("!!!!!!!!!!!!!"*20)
  print("ALL TEST COMPLETE - starting proper training")

  text_models = [
                 (DistilBert_2xInp_wrapper(max_length=256), 50),
                 (DistilBert_3xInp_wrapper(max_length=512), 50),
                 (DistilBert_2xNet_wrapper(max_length=[32, 256]), 50),
                 (DistilBert_3xNet_wrapper(max_length=[32, 256, 256]), 50),
                 ]

  # the proper training loop
  training_loop_test = False
  for text_model, batch_size in text_models:
    vision_model = EfficientTrans_wrapper()
    train_our_model(csv_file_path,
                    image_dir,
                    vision_model,
                    text_model,
                    batch_size=batch_size,
                    lr=0.0001,
                    d=d,
                    num_epochs=100,
                    max_time=3600*2,  # 2hour
                    use_mixed_precision=True,
                    training_loop_test=training_loop_test,
                    save_results=save_results)
    text_model = text_model.cpu()







