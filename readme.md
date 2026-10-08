# Mini-GPT
## Author: Mohammad Wajahath Uz Zaman

### This is the detailed Implementation Walkthroguh being documented Live, for the sake of demonstration of skills and understanding.

**I. Initial Setup:**
---
- We will create the root repository that will hold all the files for this project.
- The Architecture in plan in the initial phase is:
```text
MiniGPT/
├── data/
├── src/
├── notebooks/
├── experiments/
├── results/
└── readme.md
```
- Create `.gitkeep` in each folder so that the folder is read by git.
- Commit the architecture as our first git.

**II. Dataset Preparation:**
1. We will download the data from the url:
    - Create a python notebook in the notebooks directory and using the `requests` library, fetch the data from the url.
    - Then store it in `../data/input.txt`.
    - Inspect the data in read by checking the first 1000 characters.
2. We will now create our vocabulary:
    - There are only 64 unique characters which means our vocabulary is considerably pretty tiny.
    - Perform dataset validation and then create the vocabulary.
    - As the model is going to have character level tokenization, we can just create a set and sort it to convert it into a list.
3. Tokenizer setup:
    - We will now create two mapped lists:
    1. String to Integer where each letter from the chars list must be mapped to an integer.
    2. Integer to String where each integer must be mapped to a character/letter.
    - The above two mappings are inverse of each other.
    - Now we will Build:
    1. Encoder: This will return us an integer for the character.
    2. Decoder: This will return us a character for the string.
4. Dataset Split:
    - We will split the data into train and validation.
    - We will encode them into integers and then create tensors for each.
    - As a result we will have a large set of character ids. These are not embedding yet.
    - We will define a `get_batch` function to get the data to feed in batches.
    - Each batch will have 4 sequences/samples. And Each Sequence will be of 64 characters.
    - So now the size should be ideally --> `torch.Size([4, 64])`.
5. Now we create the Embeddings Matrices:
    - We define the embedding matrix creator using the `nn.Embedding`.
    - We convert our data into the embeddings now.
    - Now the shape of the data is:
    ```text
    Token Ids Shape:  torch.Size([4, 64])
    Token Embeddings shape:  torch.Size([4, 64, 128])
    ```
    - Now we will be needing the positional embeddings as well. We will create them simillarly but the number of embedding values will be equal to block size for position embeddings.
    - Once the position embeddings are also created we will add the two embeddings.
    - In this we will pass the positional as well as the token information into the matrix.
6. Self-Attention:
    - For self attention we require 3 more matrices: Q,K and V.
    - These three matrices are basically different representations of the same embeddings input matrix.
    - We initialize Linear layers for size of d_model.
    - The we perform the matrix multiplication to get QEmb, VEmb and KEmb.
    - Then for scores we will do:
    ```python
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_model)
    ```
    - As we are performing the `Causal Self Attention` where the model is exposed to only the current and the previous data, we will create the causal mask and implement it.
    - We now have embedding scores.
    - We will get the Attention scores by applying the softmax function.
    - Then we will multiply these matrix with value matrix to get the actual attention weights.
    - Now the shape is: `torch.Size([4, 64, 128])`.
    - At this moment we have completed the implement of single head.
    - Then we transformed it into multi head attention.
7. Transformer Block:
    - We create a transformer block as a 