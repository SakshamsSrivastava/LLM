import matplotlib.pyplot as plt  
import pandas as pd  
  
top_word_pairs = pd.read_csv('D:\\SKILL TRAINING\\projects\\LLM PROJECT\\data\\impression_300_llm.csv')  
  
# Print the column names  
print(top_word_pairs.columns)  

plt.bar(top_word_pairs['Report Name'], top_word_pairs['History'])  
plt.xlabel('Report Name')  
plt.ylabel('History')  
plt.title('Top 100 Word Pairs')  
plt.show()  
  
plt.savefig('top_100_word_pairs.png')
