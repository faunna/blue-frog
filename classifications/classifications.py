*\
    Copyright (C) 2017. faunna

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    This function is strongly based on http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
*/


def plot_decision_regions(clf, X, y, fig=None, title='', xlabel='', ylabel='', figsize=(8,6)):
  
 from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(list(reversed(['yellowgreen', 'darkseagreen', 'lightgray'])))
    cmap_bold = ListedColormap(list(reversed(['palevioletred', 'plum', 'cornflowerblue'])))
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    h = .05  # step size in the mesh
    x_min, x_max = X.values[:, 0].min() - .5, X.values[:, 0].max() + .5
    y_min, y_max = X.values[:, 1].min() - .5, X.values[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    if fig is None:
        figure = plt.figure(figsize=figsize)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.4)

        plt.scatter(X.values[:, 0], X.values[:, 1], c=y.values,
                    cmap=cmap_bold, edgecolors='k')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.xticks(())
        #plt.yticks(())
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        
    else:
        fig.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.4)

        fig.scatter(X.values[:, 0], X.values[:, 1], c=y.values,
                    cmap=cmap_bold, edgecolors='k')
        fig.set_title(title)
        fig.set_xlabel(xlabel)
        fig.set_ylabel(ylabel)
        fig.set_xticks(())
        fig.set_yticks(())
        fig.set_xlim(xx.min(), xx.max())
        fig.set_ylim(yy.min(), yy.max())
