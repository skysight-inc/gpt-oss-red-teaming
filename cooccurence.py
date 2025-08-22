import sutro as so
import json
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

results = so.get_job_results('job-xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx')

all_tags = []
for row in results.iter_rows(named=True):
    obj = json.loads(row["inference_result"])
    try:
        tags = obj["content"]["tags"]
        all_tags.append(tags)
    except:
        continue

# Get unique tags and create a mapping
unique_tags = list(set(tag for tag_list in all_tags for tag in tag_list))
tag_to_index = {tag: i for i, tag in enumerate(unique_tags)}

# Create a proper cooccurrence matrix
cooccurrence_matrix = np.zeros((len(unique_tags), len(unique_tags)))

# Count cooccurrences within each document
for tag_list in all_tags:
    for tag1, tag2 in combinations(tag_list, 2):
        idx1, idx2 = tag_to_index[tag1], tag_to_index[tag2]
        cooccurrence_matrix[idx1, idx2] += 1
        cooccurrence_matrix[idx2, idx1] += 1  # Make symmetric

# Find top cooccurring pairs
pair_counts = []
for i in range(len(unique_tags)):
    for j in range(i + 1, len(unique_tags)):
        if cooccurrence_matrix[i, j] > 0:
            pair_counts.append((cooccurrence_matrix[i, j], unique_tags[i], unique_tags[j]))

# Sort by count and get top pairs
top_pairs = sorted(pair_counts, key=lambda x: x[0], reverse=True)[:25]

# Find top cooccurring triplets
triplet_counts = Counter()
for tag_list in all_tags:
    if len(tag_list) >= 3:
        for triplet in combinations(tag_list, 3):
            triplet_counts[tuple(sorted(triplet))] += 1

top_triplets = triplet_counts.most_common(25)

print("Top 25 tag pairs:")
for count, tag1, tag2 in top_pairs:
    print(f"  {tag1} & {tag2}: {count}")

print("\nTop 25 tag triplets:")
for triplet, count in top_triplets:
    print(f"  {' & '.join(triplet)}: {count}")

write to a csv
with open("top_pairs.csv", "w") as f:
    f.write("tag_pair,count\n")
    for count, tag1, tag2 in top_pairs:
        f.write(f"{tag1} & {tag2},{count}\n")

with open("top_triplets.csv", "w") as f:
    f.write("tag_triplet,count\n")
    for triplet, count in top_triplets:
        f.write(f"{' & '.join(triplet)},{count}\n")


# create a heatmap showing only top 25 cooccurrences
all_cooccurrences = []
for i in range(len(unique_tags)):
    for j in range(i + 1, len(unique_tags)):
        if cooccurrence_matrix[i, j] > 0:
            all_cooccurrences.append((cooccurrence_matrix[i, j], unique_tags[i], unique_tags[j]))

# Sort by cooccurrence count and take top 25
top_25_cooccurrences = sorted(all_cooccurrences, key=lambda x: x[0], reverse=True)[:25]

# Get unique tags from top 25 cooccurrences
active_tags = set()
for count, tag1, tag2 in top_25_cooccurrences:
    active_tags.add(tag1)
    active_tags.add(tag2)
active_tags = sorted(list(active_tags))

# Create matrix with only active tags, but zero out non-top-50 pairs
subset_matrix = np.zeros((len(active_tags), len(active_tags)))
tag_to_active_index = {tag: i for i, tag in enumerate(active_tags)}

# Only fill in the top 50 cooccurrences
for count, tag1, tag2 in top_25_cooccurrences:
    i, j = tag_to_active_index[tag1], tag_to_active_index[tag2]
    subset_matrix[i, j] = count
    subset_matrix[j, i] = count  # Make symmetric

plt.figure(figsize=(10, 8))  # Reduced size for better saving
sns.heatmap(subset_matrix.astype(int), 
            annot=True, 
            fmt="d", 
            cmap="YlOrRd", 
            xticklabels=active_tags, 
            yticklabels=active_tags,
            cbar_kws={'label': 'Cooccurrence Count'},
            square=True)

plt.title("Heatmap: Top 25 Tag Cooccurrences")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the plot with reasonable DPI and bbox settings
plt.savefig("tag_cooccurrence_heatmap.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

