from src.models.our_model import train_our_model
import os
from src.models.model_utils import *
from torchvision.transforms import TrivialAugmentWide



if __name__ == "__main__":
   current_dir = os.path.dirname(os.path.abspath(__file__))
   food_dir = os.path.join(current_dir, '..', '..', 'data', 'processed', 'KaggleFoodDataset')
   csv_file_path = os.path.join(food_dir, 'data.csv')
   image_dir = os.path.join(food_dir, 'images')

   d = "cuda"
   save_results = True

   #running test to see that all models can run
   training_loop_tests = [True, False]
   for test in training_loop_tests:
      vision_models = [EfficientTransNonPre_wrapper(), EfficientTransNonPre_wrapper(), EfficientTrans_wrapper()]
      text_models = [DistilBert_mono_NonPre_wrapper(), DistilBert_mono_wrapper() ,DistilBert_mono_NonPre_wrapper()]
      for text_model, vision_model in zip(text_models, vision_models):
         print("\n")
         print("####"*20)
         train_our_model(csv_file_path,
                         image_dir,
                         vision_model,
                         text_model,
                         batch_size=100,
                         lr=0.0001,
                         d=d,
                         num_epochs=100,
                         use_mixed_precision=True,
                         data_aug=None,
                         max_time=3600,  # 1hour max
                         training_loop_test=test,
                         save_results=save_results)
         text_model = text_model.cpu()
         vision_model = vision_model.cpu()
         torch.cuda.empty_cache()
         torch.cuda.reset_peak_memory_stats()

      print("\n")
      print("####"*20)
      vision_model = EfficientTrans_wrapper()
      aug = TrivialAugmentWide()
      text_model = DistilBert_3xNet_wrapper()
      train_our_model(csv_file_path,
                      image_dir,
                      vision_model,
                      text_model,
                      batch_size=50,
                      lr=0.0001,
                      d=d,
                      num_epochs=50,
                      use_mixed_precision=True,
                      data_aug=aug,
                      max_time=3600 * 3,  # 1hour max
                      training_loop_test=test,
                      save_results=save_results)
      text_model = text_model.cpu()
      vision_model = vision_model.cpu()
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()

      print("\n")
      print("####"*20)
      vision_model = ConvNext_wrapper()
      aug = TrivialAugmentWide()
      text_model = DistilBert_3xNet_wrapper()
      train_our_model(csv_file_path,
                      image_dir,
                      vision_model,
                      text_model,
                      batch_size=25,
                      lr=0.0001,
                      d=d,
                      num_epochs=100,
                      use_mixed_precision=True,
                      data_aug=aug,
                      max_time=3600 * 3,  # 1hour max
                      training_loop_test=test,
                      save_results=save_results)
      text_model = text_model.cpu()
      vision_model = vision_model.cpu()
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()

      print("\n \n")
      print("!!!!!!!!!!!!!"*20)
      print("!!!!!!!!!!!!!"*20)
      print("ALL TEST COMPLETE - starting proper training \n")








