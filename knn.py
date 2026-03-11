import numpy as np
from collections import Counter
def eucladian_dist(point1,point2): #np.array is used to cast the list into numpy array 
    return np.sqrt(np.sum(np.array(point1)-np.array(point2))**2)

def knn_predict(test_point,training_points,training_labels,k):
    distances=[]
    for i in range(len(training_points)):
        dist=eucladian_dist(test_point,training_points[i])
        distances.append((dist,training_labels[i]))
    
    distances.sort(key=lambda x:x[0])#key to sort is based on the first element. 
    k_labels=[]
    for item in distances[:k]:
        distance=item[0]
        label=item[1]
        k_labels.append(label)
        
    return Counter(k_labels).most_common(1)[0][0]

#counter ({a:3 , b:2})
#most_common function returns [[a,3],[b,2]]
#here we need list @[0][0 ]  which is 'a'
#this is the whole logic behind the last line

training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [4, 5]
k = 3
print(knn_predict(test_point,training_data,training_labels,k))