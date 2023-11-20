from src.models.our_model import train_our_model
import os
from src.models.model_utils import *



if __name__ == "__main__":
  current_dir = os.path.dirname(os.path.abspath(__file__))
  food_dir = os.path.join(current_dir, '..', '..', 'data', 'processed', 'KaggleFoodDataset')
  csv_file_path = os.path.join(food_dir, 'data.csv')
  image_dir = os.path.join(food_dir, 'images')

  d = "cuda"
  text_model = DistilBert_mono_wrapper()
  save_results = True


  vision_models = [(EfficientTrans_wrapper(), 200),
                   (ViT_wrapper(), 63),
                   (ResNet_wrapper(size=18), 432),
                   (ResNet_wrapper(size=50), 144),
                   (Efficientnet_wrapper(size=4), 63),
                   ]

  #running test to see that all models can run
  training_loop_test = True
  for vision_model, batch_size in vision_models:
    text_model = DistilBert_mono_wrapper()
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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

  print("!!!!!!!!!!!!!"*20)
  print("!!!!!!!!!!!!!"*20)
  print("!!!!!!!!!!!!!"*20)
  print("ALL TEST COMPLETE - starting proper training")

  vision_models = [(ViT_wrapper(), 63),
                   (EfficientTrans_wrapper(), 180),
                   (ResNet_wrapper(size=18), 432),
                   (ResNet_wrapper(size=50), 144),
                   (Efficientnet_wrapper(size=4), 63),
                   ]

  # the proper training loop
  training_loop_test = False
  for vision_model, batch_size in vision_models:
    text_model = DistilBert_mono_wrapper()
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







