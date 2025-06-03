5 models to predict the label of an image from a given image on the yelp Photos dataset

1. Use a pretrained VGG model for transfer learning by adding custom top layers. Incorporate dense layers that take in additional non-image features (e.g., business-related metadata), concatenate them with features extracted from the VGG base, and fine-tune the model end-to-end.

2. Build a model using transfer learning with a pretrained VGG network as the base. Add custom layers on top, including dense layers that incorporate additional features (e.g., business or photo metadata), and fine-tune the entire model.

3. Use a pretrained EfficientNet model for transfer learning. Unfreeze select layers of the base network and fine-tune the model to adapt it to the target task.

4. Use a pretrained VGG model for transfer learning. Unfreeze certain layers of the base network and apply fine-tuning to adapt the model to the new dataset.

5. Initialize a model using a pretrained VGG network and train all layers from the start without freezing any part of the base model.
