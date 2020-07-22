from sklearn.mixture import GaussianMixture


x = [[ 1, 2, 3],
     [ 4, 5, 6],
     [ 7, 8, 9]]
y = [ 12, 32, 43 ]

model  = GaussianMixture(n_components = 3)
model.fit(x,y)

label = model.predict_proba([[4,5,26]])
print(label)


