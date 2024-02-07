# Hunger Games script generator
Welcome to Panem's Untold Stories ! This repository is about a Decoder-Only transformer trained over the entire Hunger Games books Saga to be able to create new stories in the style of the Author.
<br/><br/>It is really easy to use: type a first line in the input and let the magic happen!
<br/>It is a small model, inspired by [Yliess Hati](https://github.com/yliess86) and [Andrej Karpathy](https://github.com/karpathy). Due to the low performances of my laptop and the small amount of data the model is trained over,  there is a lot of room for improvement and fine-tuning.
## Stack
This project is mainly using:
- Pytorch
- Onnx and Onnxruntime web

## Training transcript
The text for the training dataset : [The Hunger Games - Trilogy](https://www.deyeshigh.co.uk/downloads/literacy/world_book_day/the_hunger_games_-_trilogy.pdf), a pdf that i then treated to remove unnecessary elements, symbols and other stuff that would affect the training.

I also included the txt file with the treated raw data if you want to play with it at home, and a Jupyter Notebook to test the model directly, and to tweak around things to make sure its working 

I hope you'll enjoy it, and have fun !
<br/>
![Enjoy !](https://media.giphy.com/media/BsH0I7tBbP03e/giphy.gif)
