import pandas as pd


def normalize(s: str) -> str:
    '''
    Doc:
    -------------------------------------------------------------
    Arguments: The string you want to "normalize".
    -------------------------------------------------------------
    Usage:
    Insert a string you want to normalize and do the following changes:
    á -> a
    é -> e
    í -> i
    ó -> o
    ú -> u
    ñ -> ny
    -------------------------------------------------------------
    Return:
    A word with the above changes.


    -------------------------------------------------------------
    '''
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ñ", "ny")
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s


def normalize_columns(data: pd.DataFrame, inplace: bool) -> pd.DataFrame:
    '''
    Doc:
    -------------------------------------------------------------
    Arguments:
    Data:= The pandas DataFrame you want to "normalize".
    Inplace:= A Boolean to set if you want to do the changes inplace or not.
    -------------------------------------------------------------
    Usage:
    Insert pandas DataFrame you want to normalize columns:
    Changes that will be made in the names
    á -> a -> A
    é -> e -> E
    í -> i -> I
    ó -> o -> O
    ú -> u -> U
    ñ -> ny -> NY
    -------------------------------------------------------------
    Return:
    A pandas DataFrame with the columns normalized.


    -------------------------------------------------------------
    '''
    Columnas_normalizadas = [normalize(x).upper() for x in data.columns]
    Columnas_lectura = [x for x in data.columns]
    renombrar_columnas = dict(zip(Columnas_lectura, Columnas_normalizadas))
    return data.rename(columns=renombrar_columnas, inplace=inplace)


def binarizar_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Doc: Este método transformará todas las columnas de un dataframe
    que contengan Sí y No.
    -------------------------------------------------------------
    Arguments: The DataFrame you want to binarize.
    -------------------------------------------------------------
    Usage:
    if the word starts with 'S' or 's' then turn it to 1.
    elif the word starts with 'N' or 'n' then turn it to 0.
    else return the word.
    -------------------------------------------------------------
    Return:
    A pandas DataFrame with the above changes.
    -------------------------------------------------------------

    """
    def auxiliar_binarizar(x: str) -> str:
        if x.startswith('N') or x.startswith('n'):
            return 0
        elif x.startswith('S') or x.startswith('s'):
            return 1
        else:
            return x

    variables_binarizar = [x for x in data.select_dtypes(
        include=['object', 'category']).columns]
    data[variables_binarizar] = data[variables_binarizar].applymap(
        lambda x: auxiliar_binarizar(x))
    return data

def dibujar_curva_roc(modelo, nombre_modelo, X_test, y_test):
    # importamos librerías
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    
    
    # fijamos estilo del plot
    plt.style.use('seaborn-whitegrid')
    
    # predicciones de probabilidad
    y_prob_logr = modelo.predict_proba(X_test)[:,1]
    # estimaciones de la curva ROC
    fpr_logr, tpr_logr, thresholds_logr = roc_curve(y_test, y_prob_logr)
    
    plt.figure(figsize=(8, 6)) # tamaño de la imagen
    plt.title(f'Curva ROC {nombre_modelo}') # título de la imagen
    plt.plot(fpr_logr, tpr_logr, label=nombre_modelo) # curva roc y label del modelo
    
    plt.plot([0, 1], [0, 1], 'g--') # clasificador trivial
    plt.plot([0, 1], [0, 0], 'k') # linea negra inferior
    plt.plot([1, 1], [0, 1], 'k') # línea negra derecha
    plt.axis([-0.05, 1.05, -0.05, 1.05]) # tamaño de los ejes
    # plt.axis('equal')
    plt.grid(True) # para mostrar las líneas del grid
    plt.text(x = 0.6, y = 0.05, s= f'AUC {nombre_modelo}: {round(roc_auc_score(y_test, y_prob_logr),2)}', fontsize=12) #texto AUC
    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=14) # xlabel
    plt.ylabel('True Positive Rate (Recall)', fontsize=14)    # ylabel
    
    plt.legend() # mostrar leyenda
    plt.show() # sacar imagen
    
def comparar_modelos_ROC(modelo1, nombre_modelo_1, modelo2, nombre_modelo_2, X_test, y_test):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    plt.style.use('seaborn-whitegrid')

    y_prob_logr = modelo1.predict_proba(X_test)[:,1]
    y_prob_tree = modelo2.predict_proba(X_test)[:,1]
    
    fpr_logr, tpr_logr, thresholds_logr = roc_curve(y_test, y_prob_logr)
    fpr_tree, tpr_tree, thresholds_tre = roc_curve(y_test, y_prob_tree)
    
    plt.figure(figsize=(8, 6))
    plt.title('Comparación Modelos mediante ROC', )
    plt.plot(fpr_logr, tpr_logr, label=nombre_modelo_1)
    plt.plot(fpr_tree, tpr_tree, label=nombre_modelo_2)
    
    plt.plot([0, 1], [0, 1], 'g--')
    plt.plot([0, 1], [0, 0], 'k')
    plt.plot([1, 1], [0, 1], 'k')
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    # plt.axis('equal')
    plt.grid(True)
    
    plt.text(x = 0.6, y = 0.05, s= f'AUC {nombre_modelo_2}: {round(roc_auc_score(y_test, y_prob_tree),2)}', fontsize=12)
    plt.text(x = 0.6, y = 0.1, s= f'AUC {nombre_modelo_1}: {round(roc_auc_score(y_test, y_prob_logr),2)}', fontsize=12)
    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=14) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=14)    # Not shown
    
    plt.legend()
    plt.show()


from sklearn import metrics 
import seaborn as sns
import matplotlib.pyplot as plt


def pr_auc_score(clf, x, y):
    '''
        This function computes area under the precision-recall curve. 
    '''
      
    precisions, recalls,_ = precision_recall_curve(y, clf.predict_proba(x)[:,1], pos_label=1)
    
    return auc(recalls, precisions)


from sklearn.model_selection import StratifiedKFold
def imbalanced_cross_validation_score(clf, x, y, cv, scoring, sampler):
    '''
        This function computes the cross-validation score of a given 
        classifier using a choice of sampling function to mitigate 
        the class imbalance, and stratified k-fold sampling.
        
        The first five arguments are the same as 
        sklearn.model_selection.cross_val_score.
        
        - clf.predict_proba(x) returns class label probabilities
        - clf.fit(x,y) trains the model
        
        - x = data
        
        - y = labels
        
        - cv = the number of folds in the cross validation
        
        - scoring(classifier, x, y) returns a float
        
        The last argument is a choice of random sampler: an object 
        similar to the sampler objects available from the python 
        package imbalanced-learn. In particular, this 
        object needs to have the method:
        
        sampler.fit_sample(x,y)
        
        See http://contrib.scikit-learn.org/imbalanced-learn/
        for more details and examples of other sampling objects 
        available.  
    
    '''
    
    cv_score = 0.
    train_score = 0.
    test_score = 0.
    
    # stratified k-fold creates folds with the same ratio of positive 
    # and negative samples as the entire dataset.
    
    skf = StratifiedKFold(n_splits=cv, random_state=None, shuffle=False)
    
    for train_idx, test_idx in skf.split(x,y):
        
        xfold_train_sampled, yfold_train_sampled = sampler.fit_resample(x[train_idx],y[train_idx])
        clf.fit(xfold_train_sampled, yfold_train_sampled)
        
        train_score = scoring(clf, xfold_train_sampled, yfold_train_sampled)
        test_score  = scoring(clf, x[test_idx], y[test_idx])
        
        print("Train AUPRC: %.2f Test AUPRC: %.2f"%(train_score,test_score))

        cv_score += test_score
        
    return cv_score/cv


def show_results_1(y_true, y_pred):
    """
    Doc: Con el vector de predicciones y real, devuelve varias métricas.
    -------------------------------------------------------------
    Arguments: 
    y_true := the real vector.
    y_pred := the predictions.
    -------------------------------------------------------------
    Usage:

    -------------------------------------------------------------
    Return:
    A print with the results.
    -------------------------------------------------------------

    """
    c_mat = metrics.confusion_matrix(y_true,y_pred)
    sns.heatmap(c_mat, square=True, annot=True, fmt='d', cbar=True, cmap=plt.cm.Blues)
    plt.ylabel('Clase real')
    plt.xlabel('Predicción');
    plt.gca().set_ylim(2.0, 0)
    plt.show()
    print("Resultados: ")
    print(f'\taccuracy: {metrics.accuracy_score(y_true=y_true, y_pred=y_pred):.3f}')
    print(f'\trecall: {metrics.recall_score(y_true=y_true, y_pred=y_pred):.3f}')
    print(f'\tprecision: {metrics.precision_score(y_true=y_true, y_pred=y_pred):.3f}')
    print(f'\tf1_score: {metrics.f1_score(y_true=y_true, y_pred=y_pred):.3f}')
    

import plotly.graph_objects as go

def plot_2d(component1, component2):

    fig = go.Figure(data=go.Scatter(
        x=component1,
        y=component2,
        mode='markers',
        marker=dict(
            size=20,
            color=y,  # set color equal to a variable
            colorscale='Rainbow',  # one of plotly colorscales
            showscale=True,
            line_width=1
        )
    ))
    fig.update_layout(margin=dict(l=100, r=100, b=100,
                      t=100), width=1000, height=600)
    fig.layout.template = 'plotly_dark'

    fig.show()


def plot_3d(component1, component2, component3):
    fig = go.Figure(data=[go.Scatter3d(
        x=component1,
        y=component2,
        z=component3,
        mode='markers',
        marker=dict(
            size=10,
            color=y,                # set color to an array/list of desired values
            colorscale='Rainbow',   # choose a colorscale
            opacity=1,
            line_width=1
        )
    )])
 # tight layout
    fig.update_layout(margin=dict(l=50, r=50, b=50, t=50),
                      width=900, height=500)
    fig.layout.template = 'plotly_dark'

    fig.show()


def initial_eda(df):
    if isinstance(df, pd.DataFrame):
        total_na = df.isna().sum().sum()
        print("Dimensions : %d rows, %d columns" % (df.shape[0], df.shape[1]))
        print("Total NA Values : %d " % (total_na))
        print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))
        col_name = df.columns
        dtyp = df.dtypes
        uniq = df.nunique()
        na_val = df.isna().sum()
        for i in range(len(df.columns)):
            print("%38s %10s   %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i]))
        
    else:
        print("Expect a DataFrame but got a %15s" % (type(df)))
        
        
def find_categoricals(df:pd.DataFrame) -> list:
    categorical = [var for var in df.columns if df[var].dtype=='O']

    print('There are {} categorical variables\n'.format(len(categorical)))

    print('The categorical variables are :\n\n', categorical)
    
    return categorical

# def listar_valores_categoricos(df: pd.DataFrame):
#     categorical = [var for var in df.columns if df[var].dtype=='O']
    
#     for var in categorical: 
    
#     print(f'{df[var].value_counts()}, Pje: {round(df[var].value_counts()/np.float(len(df))*100,2)} %')
    
    








   
    

