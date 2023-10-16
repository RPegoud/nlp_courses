# ***Build your own NLP Project***

This in your occasion to work on an NLP project from the **exploratory data analysis** (EDA) **to the inference phase**, on a topic of your choice. Don't be intimadated by the number of steps, there's almost nothing new compared to previous assignments!

## ***1. Select a dataset***

Find a text dataset on the topic you're interested in, websites like [**Kaggle**](https://www.kaggle.com/datasets) provide ready-to-use datasets. You can choose a task we've already solved in class (**classification** or **sentiment analysis**) or a new one if you feel ready!

## ***2. Exploratory data analysis $⇾$ jupyter notebook***

Start by exploring your data, examine some samples, plot the information that you deem relevant, identify potential features you can engineer to improve performance later ...

## ***3. Build your preprocessing pipeline $⇾$ python script***

Similarly to the first lab, assemble a preprocessing script. Make sure to explain your design choices and why you use specific functions in a particular order. You can also add any utility function that you might need later to this script.

## ***4. Train a baseline model $⇾$ jupyter notebook***

Import your preprocessing pipeline, apply it to your dataset and train the machine learning model of your choice (sklearn or similar) without any particular parameter tuning or feature engineering. The goal here is simply to obtain a baseline model which we'll use as reference for future experiments. This is the good moment to create a model class (remember lab 2) that will facilitate iteration later on.

## ***5. Improve on the baseline $⇾$ jupyter notebook***

Using techniques of your choice, improve on the baseline results. This is the moment to demonstrate your ability to identify the bottlenecks in your training process and potential problems with your model / data.

Here are some common ideas to get you started:

* Balancing classes (oversampling or subsampling)
* Applying class weights
* Monitoring training curves
* Early stopping
* Hyperparameter tuning, grid search, random search
* Or anything else you deem coherent

## ***6. Use Tensorflow (or PyTorch, JAX, ...) and train a sequence model of your choice (RNN, GRU, LSTM, Transformer, ...) $⇾$ jupyter notebook***

Finally, we want to further improve the performance of our model by applying our deep learning skills to the problem. You can either build and train a model from scratch or finetune a pre-trained model (transfer learning, you can find models to download on websites like [***HuggingFace***](https://huggingface.co/models)). Make sure to comment on the architecture you decide to use!

## ***7. Publish your project on GitHub and add a nice README.md file***

Now that your project is complete, publish it on your GitHub and add a README.md file (it will be used as the landing page of your repository). This readme should contain:

* A global description of your project
* A link to your dataset
* A table containing the performances of each model you implemented
* Instruction on how to install and run your project (you can use poetry, anaconda, or a requirements.txt file)
* A list of references that you used (coding tutorials, research papers, other documents ...)

## ***⚠️ Grades ⚠️***

What I'm looking forward to see in your projects:

* Justification of your design choices and **thorough analysis** of your results/plots:
  * Simply saying *"model B is better than model A because it has better precision"* is not enough
  * In an interview, you'll be asked the simple and deadly question *"Why is that?"*, anticipate it and explain why your design decision make sense, what weaknesses they have and **demonstrate your understanding of the models you use**
  
* **Do not shy away from difficult problems**, the results of your model will not influence your grade. I'm looking for senseful and justified decisions more than 99% f1 score on a standard task. In the industry, most problems won't be solved by throwing a random machine learning model at a dataset, use this project as a way to test different approaches and experiment!
* **Clean code** (for example, define preprocessing functions in a separate Python script, use classes as in TP2, use MarkDown in your notebooks to explain your process etc ...)
  
* General **presentation of the GitHub repository** (remember that the goal is to show it to employers, make it nice and easy to see what you've been working on)
  
* You may use **ChatGPT** for **references** and **ideas** (make sure to **mention** what it helped you to understand or the ideas it gave you), but **copy/pasting prompts will lead to a refit** (and having hard times during interviews, you get the idea). The same goes for copying notebooks and code you found on Kaggle of course.

Here's an example of a project done last year in the ML class:

<div align="center">
  <a href="https://github.com/RPegoud/PyTorch_traffic_sign_detection"><img src="https://gh-card.dev/repos/RPegoud/PyTorch_traffic_sign_detection.svg"></a>
</div>
