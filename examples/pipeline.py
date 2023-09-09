from transformers import pipeline

out = pipeline("sentiment-analysis")
input_text = ["we love u", "we hate u", "good morning", "climate is bad"]
for i in input_text:
    print("input is ", i)
    print("output is ", out(i))

out = pipeline("text-generation", model="gpt2")
input_text = ["I dream of", "where are"]
for i in input_text:
    print("input is ", i)
    print("output is ", out(i, max_length=30, num_return_sequences=1))

out = pipeline("text2text-generation")
input_text = ["translate from English to French: I'm very happy",
              "question: how is the weather ? context: sunny and windy"]
for i in input_text:
    print("input is ", i)
    print("output is ", out(i))
