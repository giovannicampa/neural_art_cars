# Neural Art Cars
Inspired by the BMW's art car series [1], this repo explores hat an art car could look like.

## Procedure

- Fast neural style transfer is applied on the whole original image [2].
- A pretrained net is used to perform a semantic segmentation and find the pixels that represent the car [3].
- The car's pixels of the original image are selected and, with a thresholding operation [4], the dark areas are found. These are saved in a separate mask.
- The pixels of the stylised image that do not belong to the car, or do belong to the dark areas of the car are selected. They are substituted with the corresponding ones from the original image.

The result shows an image in which the style is only applied to the car.


| Semantic segmentation | Dark car areas | Pixels to keep from original |
|:-------------------------:|:-------------------------:|:-------------------------:|
| Finding the pixels that belong to the car | Of the car's pixels, select the dark ones through thresholding | Intersection between the previous two masks |
|<img width="1604" src="https://github.com/giovannicampa/neural_art_cars/blob/master/images/masks/mask_car.png"> | <img width="1604" src="https://github.com/giovannicampa/neural_art_cars/blob/master/images/masks/mask_dark_areas.png"> | <img width="1604" src="https://github.com/giovannicampa/neural_art_cars/blob/master/images/masks/pixels_from_original.png"> |

## Results

The table below shows an example of how the art gets applied to the whole car. The last image shows the step of the thresholding.
The artwork used is by Bauhaus member Artist Fritz Kuhr.

| Art | Stylised Car | Art Car |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/giovannicampa/neural_art_cars/blob/master/images/styles/bauhaus.jpeg"> | <img width="1604" src="https://github.com/giovannicampa/neural_art_cars/blob/master/images/output/bauhaus_car_complete.png"> | <img width="1604" src="https://github.com/giovannicampa/neural_art_cars/blob/master/images/output/bauhaus_car.png"> |

The table below shows furhter examples of art cars.

| Name | Art | Art Car |
|:-------------------------:|:-------------------------:|:-------------------------:|
| Composition VII, 1913 - Wassily Kandinsky | <img width="800" src="https://github.com/giovannicampa/neural_art_cars/blob/master/images/styles/kandinsky.jpeg"> | <img width="800" src="https://github.com/giovannicampa/neural_art_cars/blob/master/images/output/kandinsky_car.png"> |
| Femme au béret et à la robe quadrillée, 1937 - Pablo Picasso |<img width="800" src="https://github.com/giovannicampa/neural_art_cars/blob/master/images/styles/picasso.jpeg"> | <img width="800" src="https://github.com/giovannicampa/neural_art_cars/blob/master/images/output/picasso_car.png"> |


## References
- [1] https://www.artcar.bmwgroup.com/en/art-car/
- [2] https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2
- [3] https://github.com/divamgupta/image-segmentation-keras/blob/master/README.md
- [4] https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html
