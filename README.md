## Dataset
* The Shenzhen dataset consists of 662 frontal CXR images, in which 326 images are classified as normal and 336 images are classified as manifestation of tuberculosis.
* The Montgomery dataset contains 138 frontal chest X-rays, of which 80 images are classified as normal cases and 58 images are with manifestations of tuberculosis.
* The Belarus dataset is made of 304 CXR images of patients with confirmed TB.
* Covid-19 Chest X-Ray Database consists of 3616 COVID-19 positive cases, 10,192 normal, 6012 Lung Opacity (Non-COVID lung infection) and 1345 Viral Pneumonia images.

### From the above 4 public datasets, I merged and populated a dataset consisting of `22,269` images and then split them into `training`/`validation`/`testing` set with ratio `80% - 10% - 10%.`


| Type          | COVID-19 | Lung Opacity | Normal | Pneunomia | Tuberculosis | Total |
| :-            | :-:      | :-:          | :-:    | :-:       | :-:          | :-:   | 
| Train         | 2,892    | 4,809        | 8,478  | 1,076     | 558          | 17,813|
| Validation    | 362      | 601          | 1,060  | 134       | 70           | 2,227 |
| Test          | 362      | 602          | 1,060  | 135       | 70           | 2,229 |
| Total         | 3,616    | 6,012        | 10,598 | 1,345     | 698          | 22,269|

----------------------------------------

## Accuracy on Testset
| ResNet50 | ResNeXt101| Swin Transformer | EfficientNet B2 | Ensemble |
|---|---|---|---|---|
| 92.059% | 95.110% | 93.585% | 95.155% | 95.289% |

## Ensemble Result
![Ensemble Result](/Result/ensemble.PNG)

----------------------------------------

## Confusion Matrix

### Swin Transformer
![swin_transformer_confusion_matrix](/Confusion_Matrix/swin_transformer.PNG)

### EfficientNet B2
![efficientnet_b2_confusion_matrix](/Confusion_Matrix/effnet_b2.PNG)