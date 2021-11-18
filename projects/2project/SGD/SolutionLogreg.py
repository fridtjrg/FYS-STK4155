import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
sns.set()

#====================== DATA
from SDG import SDG
import sys
sys.path.append("../Data")


from DataClassification import X_test, X_train, Y_train_onehot, Y_test_onehot, accuracy_score_numpy, X, y_test, y_train
z_test = y_test
z_train = y_train

n_hidden_neurons = 50
batch_size = 50
epochs = 100

Eta = np.logspace(-5, -2, 5)
Lambda = np.logspace(-5, -2, 5)

accuracy_own = np.empty((5,5))
accuracy_sk = np.empty((5,5))

for i, eta in enumerate(Eta):
    for j, _lambda in enumerate(Lambda):

        #===============================#
        #           Training            #
        #===============================#

        #------ own
        sdg = SDG(learning_rate=eta, n_epochs=epochs, batch_size=batch_size, method='logreg', lmbda= _lambda)
        beta = sdg.train(X_train, z_train)

        #------ sklearn
        sdg_sk = SGDClassifier(loss='log', alpha=_lambda, max_iter=epochs, learning_rate='constant', eta0=eta)
        sdg_sk.fit(X_train, z_train)
        
        #===============================#
        #           Testing             #
        #===============================#

        acc = sdg.logreg_accuracy(X_test, z_test, beta)
        accuracy_own[i,j] = acc

        acc_sk = sdg_sk.score(X_test,z_test)
        accuracy_sk[i,j] = acc_sk


fig1, ax1 = plt.subplots(figsize=(7, 5))
sns.heatmap(accuracy_own, annot=True, ax=ax1, cmap="viridis")
#ax.set_title("Test ridge MSE")
ax1.set_ylabel("$\eta$")
ax1.set_xlabel("$\lambda$")
ax1.set_xticklabels(["{:.1e}".format(l) for l in Lambda])
ax1.set_yticklabels(["{:.1e}".format(e) for e in Eta])

plt.subplots_adjust(
top=0.98,
bottom=0.117,
left=0.08,
right=1,
hspace=0.2,
wspace=0.2
)
plt.savefig('../Figures/GD/SGD_test_heatmap_logreg_own.pdf')

fig2, ax2 = plt.subplots(figsize=(7, 5))
sns.heatmap(accuracy_sk, annot=True, ax=ax2, cmap="viridis")
#ax.set_title("Test ridge MSE")
ax2.set_ylabel("$\eta$")
ax2.set_xlabel("$\lambda$")
ax2.set_xticklabels(["{:.1e}".format(l) for l in Lambda])
ax2.set_yticklabels(["{:.1e}".format(e) for e in Eta])

plt.subplots_adjust(
top=0.98,
bottom=0.117,
left=0.08,
right=1,
hspace=0.2,
wspace=0.2
)
plt.savefig('../Figures/GD/SGD_test_heatmap_logreg_sk.pdf')
plt.show()