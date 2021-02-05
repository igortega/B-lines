"""

Main script

"""
import matplotlib.pyplot as plt
import numpy as np
from frame_extraction import extract_all
from cluster import cluster_all
from polar import polar_all
from regression import train_test_split
from scoring import score_all
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix

"extract_all() to extract and crop frames of all videos"

"cluster_all() to calculate centroids of all videos"

"polar_all() to reshape centroids of all videos to polar format"

"train_test_split() to create stratified train and test sets"

# Split train and test sets
train, test = train_test_split()

# Calculate scores of train set
train_scores = score_all(train['Id'])
# Labels of train set
train_labels = np.array(train['B-lines'])

# Train model
reg = LogisticRegression()
reg.fit(train_scores, train_labels)

# Calculate scores of test set
test_scores = score_all(test['Id'])

# Predict labels of test set
predict_labels = reg.predict(test_scores)

# Compare prediction to true labels
test_labels = np.array(test['B-lines'])

# Evaluate precission
correct_predictions = sum(predict_labels == test_labels)
print(correct_predictions, "correct predictions out of", len(test_labels))
print(correct_predictions/len(test_labels)*100, "percent accuracy")

" figures() to generate graphs, etc."

# FIGURES

# Scores 1 and 2
plt.scatter(train_scores[train_labels == 1][:,0],train_scores[train_labels==1][:,1], alpha=0.9, label='B-lines')
plt.scatter(train_scores[train_labels == 0][:,0],train_scores[train_labels==0][:,1], alpha=0.8, label='No B-lines')
plt.xlabel('Total mean')
plt.ylabel('Quarter mean')
plt.legend()
plt.savefig('figures/scores1_2.png')
plt.show()

# Scores 1 and 3
plt.scatter(train_scores[train_labels == 1][:,0],train_scores[train_labels==1][:,2], alpha=0.9, label='B-lines')
plt.scatter(train_scores[train_labels == 0][:,0],train_scores[train_labels==0][:,2], alpha=0.8, label='No B-lines')
plt.xlabel('Total mean')
plt.ylabel('Fraction above half max.')
plt.legend()
plt.savefig('figures/scores1_3.png')
plt.show()

# Scores 2 and 3
plt.scatter(train_scores[train_labels == 1][:,1],train_scores[train_labels==1][:,2], alpha=0.9, label='B-lines')
plt.scatter(train_scores[train_labels == 0][:,1],train_scores[train_labels==0][:,2], alpha=0.8, label='No B-lines')
plt.xlabel('Quarter mean')
plt.ylabel('Fraction above half max.')
plt.legend()
plt.savefig('figures/scores2_3.png')
plt.show()


# Plot test
# Scores 1 and 2
# True positives
TP = (test_labels == 1) & (predict_labels == 1)
plt.scatter(test_scores[TP][:,0],test_scores[TP][:,1], label='True positive', marker='*', color='tab:blue')
# True negatives
TN = (test_labels == 0) & (predict_labels == 0)
plt.scatter(test_scores[TN][:,0],test_scores[TN][:,1], label='True negative', color='tab:orange')
# False positives
FP = (test_labels == 0) & (predict_labels == 1)
plt.scatter(test_scores[FP][:,0],test_scores[FP][:,1], label='False positive', marker='*', color='tab:orange')
# False negatives
FN = (test_labels == 1) & (predict_labels == 0)
plt.scatter(test_scores[FN][:,0],test_scores[FN][:,1], label='False negative', color='tab:blue')

plt.xlabel('Total mean')
plt.ylabel('Quarter mean')
plt.legend()
plt.savefig('figures/discussion.png')
plt.show()



# #TEST
# # Scores 1 and 2
# plt.scatter(test_scores[test_labels == 1][:,0],test_scores[test_labels==1][:,1], alpha=0.9, label='B-lines')
# plt.scatter(test_scores[test_labels == 0][:,0],test_scores[test_labels==0][:,1], alpha=0.8, label='No B-lines')
# plt.xlabel('Total mean')
# plt.ylabel('Quarter mean')
# plt.legend()
# plt.savefig('figures/scores1_2.png')
# plt.show()

# # Scores 1 and 3
# plt.scatter(test_scores[test_labels == 1][:,0],test_scores[test_labels==1][:,2], alpha=0.9, label='B-lines')
# plt.scatter(test_scores[test_labels == 0][:,0],test_scores[test_labels==0][:,2], alpha=0.8, label='No B-lines')
# plt.xlabel('Total mean')
# plt.ylabel('Fraction above half max.')
# plt.legend()
# plt.savefig('figures/scores1_3.png')
# plt.show()

# # Scores 2 and 3
# plt.scatter(test_scores[test_labels == 1][:,1],test_scores[test_labels==1][:,2], alpha=0.9, label='B-lines')
# plt.scatter(test_scores[test_labels == 0][:,1],test_scores[test_labels==0][:,2], alpha=0.8, label='No B-lines')
# plt.xlabel('Quarter mean')
# plt.ylabel('Fraction above half max.')
# plt.legend()
# plt.savefig('figures/scores2_3.png')
# plt.show()


