#Code courbes ROC (regression logistique):

 

import numpy as np

import pandas as pd

import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import metrics

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve

#from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report

import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

from pandas import ExcelWriter

 

File = "C:\\Users\\nassd\\Downloads\\Oedème_Req.xls"

df = pd.read_excel(File)

df.head()

rows = len(df.axes[0])

cols = len(df.axes[1])

i = 2

while (i < cols):

    med = df[df.columns[i]].median()

    df[df.columns[i]] = df[df.columns[i]].fillna(med)

    i = i + 1

print('\n')

print ('Analyse du fichier Oedème Req Req')

print('Nombre enregistrements : ',rows) #nombre d'enregistrements

print('Nombre de variables : ',cols) #nombre de colonnes

print('Variables :\n',df.columns) #noms, type et nombre de colonnes

stats = df.describe()

#writer = ExcelWriter('ROC Oedème Stats.xlsx')

#stats.to_excel(writer, 'Feuille1')

#writer.save()

 

#Scikit-learn: valeurs caractéristiques et valeur cible

y = df['OEDEME']

print('Régression logistique. Variable prédite: OEDEME')

print('Nbre occurrences :\n',y.value_counts()) #nbre d’occurrences des différentes valeurs de la variable

 

# Créer le fichier Excel des résultats

dff = pd.DataFrame(columns = ['Variable prédite', 'Variable explicative',

                              'Test U1 de Mann-Whitney sur la totalité des données',

                              'p', 'Med Path', 'Med T', 'Moy Path', 'Moy T', 'SD Path', 'SD T',

                              'AUC (courbe ROC, train set)',                              

                              'p train', 'Stats Path train', 'Stats T train', 'Seuil de la variable',

                              'AUC (courbe ROC, test set)',

                              'Se (%)', 'Sp (%)', 'Exactitude',                         

                              'Coefficient', 'Constante du modèle',

                              'Précision du modèle (VPP, %)',

                              'Meilleur seuil', 'Meilleur F1-Score',

                              'VPP (%)', 'VPN (%)', 'Indice de Youden',

                              'VN', 'VP', 'FN', 'FP'],

                   index = ['Age', 'CBI', 'SSI','Pachy', 'L1', 'L2', 'V1', 'V2', 'PD',

                            'Radius', 'DA', 'IOPnct', 'bIOP', 'DA Ratio', 'IR',

                            'ARTh', 'SPA-A1', 'CBI_SYM', 'SSI_SYM','Pachy_SYM', 'L1_SYM', 'L2_SYM',

                            'V1_SYM', 'V2_SYM', 'PD_SYM','Radius_SYM', 'DA_SYM', 'IOPnct_SYM',

                            'bIOP_SYM', 'DA_Ratio_SYM', 'IR_SYM','ARTh_SYM', 'SPA-A1_SYM',

                            'AllParameters'])

 

for i in range(cols - 1):

    Param = df.columns[i+1]

 

    # Analyse Param

    print('\n', 'Analyse de ', Param)

    U1, p = mannwhitneyu(df.query('OEDEME == 1')[[Param]],

                         df.query('OEDEME == 0')[[Param]], alternative='two-sided',

                         axis=0, method='auto')

    MoyKC = df.query('OEDEME == 1')[[Param]].mean()

    MoyT = df.query('OEDEME == 0')[[Param]].mean()

    MedKC = df.query('OEDEME == 1')[[Param]].median()

    MedT = df.query('OEDEME == 0')[[Param]].median()

    SDKC = df.query('OEDEME == 1')[[Param]].std()

    SDT = df.query('OEDEME == 0')[[Param]].std()

   

    x = df[[Param]]

    # Fractionner le dataset en un set d’apprentissage (70%) et un set d’évaluation (30%)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    for train_index, test_index in split.split(df, df['OEDEME']):

        train_set = df.loc[train_index]

        test_set = df.loc[test_index]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)

    print('Dataframe séparé en un train set et un test set')

    dft = df.loc[train_index]

    dfs = df.loc[test_index]

    print('Train set:', '\n', dft['OEDEME'].value_counts(normalize = True))

    print('Test set:', '\n', dfs['OEDEME'].value_counts(normalize = True))

    Utrain, ptrain = mannwhitneyu(dft.query('OEDEME == 1')[[Param]],

                         dft.query('OEDEME == 0')[[Param]], alternative='two-sided',

                         axis=0, method='auto')

    MoyPatht = dft.query('OEDEME == 1')[[Param]].mean()

    MoyTt = dft.query('OEDEME == 0')[[Param]].mean()

    SDPatht = dft.query('OEDEME == 1')[[Param]].std()

    SDTt = dft.query('OEDEME == 0')[[Param]].std()

    M1 = round(MoyPatht[0], 2)

    M2 = round(MoyTt[0], 2)

    M3 = round(SDPatht[0], 2)

    M4 = round(SDTt[0], 2)

    StatsPathtrain = str(M1) + "+" + str(M3)

    StatsTtrain = str(M2) + "+" + str(M4)

    print(StatsPathtrain)

    print(StatsTtrain)

    #instanciation du modèle:

    modele_regLog = LogisticRegression(penalty=None,solver='newton-cg')

    #training

    modele_regLog.fit(x_train,y_train)

    #affichage des coefficients et de la constante

    print(pd.DataFrame({"var":x_train.columns,"coef":modele_regLog.coef_[0]}))

    print('Constante du modèle: ', modele_regLog.intercept_)

    #précision du modèle (VPP)

    precision = modele_regLog.score(x_test,y_test)

    print('Précision du modèle dans le test set (Valeur Prédictive Positive, %): ', precision*100)

    print('\n')

   

    # Définir les metrics

    y_pred_proba = modele_regLog.predict_proba(x_train)[::,1]

    fpr, tpr,_=metrics.roc_curve(y_train, y_pred_proba)

    auctrain = metrics.roc_auc_score(y_train, y_pred_proba)

    print('AUC (aire sous la courbe ROC du train set): ',str(auctrain))

   

    # Créer la courbe ROC, le scatter-plot des données et des probas du train set

    plt.plot(fpr,tpr,label= str(Param)+" AUC (train set)="+str(auctrain))

    plt.ylabel("True Positive Rate")

    plt.xlabel("False Positive Rate")

    plt.plot([0,1], [0,1], color = 'navy', linestyle = '--')

    plt.legend(loc=4)

    plt.show()

    print('\n')

    plt.plot(y_train,x_train,"ob") # ob = type de points "o" ronds, "b" bleus

    plt.xlabel("OEDEME (y_train)")

    plt.ylabel(str(Param)+" (x_train)")

    plt.show()

    print('\n')

    plt.plot(x_train, y_pred_proba,"ob")

    plt.ylabel("Probabilité de OEDEME (y_pred_proba)")

    plt.xlabel(str(Param)+" (x_train)")

    plt.show()

    prediction_KC = modele_regLog.predict([[1.0]])

    print('KC prédit pour ', Param,' = 0,0: ', prediction_KC, modele_regLog.predict_proba([[0.0]]))

    prediction_KC = modele_regLog.predict([[1.0]])

    print('KC prédit pour ', Param, ' = 1,0: ', prediction_KC, modele_regLog.predict_proba([[0.9]]))

    precisions, rappels, seuils = precision_recall_curve(y_train, y_pred_proba)

    precisions = precisions[:-1]

    rappels = rappels[:-1]

    condition = rappels + precisions > 0

    precisions = precisions[condition]

    rappels = rappels[condition]

    seuils = seuils[condition]

    f1_scores = 2 * rappels * precisions/(rappels + precisions)

    seuil_f1_max = np.max(f1_scores)

    print('np.max(f1_scores): ', seuil_f1_max)

    val_seuil_bis = seuils[np.argmax(f1_scores)]

    print('val_seuil_bis (train set) : ', val_seuil_bis)

    un_moins_spec, sens, seuil = roc_curve(y_train, y_pred_proba)

    recall_0 = 1 - un_moins_spec[sum(seuil>val_seuil_bis)]

    print("Sp(Recall 0, train set, %): ", float(recall_0*100))

    recall_1 = sens[sum(seuil>val_seuil_bis)]

    print("Se (Recall 1, train set, %): ", float(recall_1*100))

    youden = recall_0 + recall_1 - 1

    print('Indice de Youden: ', youden)

    seuil_sur_ypred=seuils[np.argmax(f1_scores)]

    dfpourseuil=pd.DataFrame({'xtrain':x_train[Param],'ypred':y_pred_proba})

    seuil_sur_f1=min(dfpourseuil[dfpourseuil['ypred']>seuil_sur_ypred]['xtrain'])

    df_closest = dfpourseuil.iloc[(dfpourseuil['ypred']-0.5).abs().argsort()[:1]]

    seuil_proba_50 = df_closest.iloc[0,0]

    print('Seuil ', Param, ':', seuil_sur_f1)

    print('\n')

   

    # Définir les metrics

    y_pred_proba = modele_regLog.predict_proba(x_test)[::,1]

    fpr, tpr,_=metrics.roc_curve(y_test, y_pred_proba)

    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    print('AUC (aire sous la courbe ROC du test set): ',str(auc))

    print('\n')

    precisions, rappels, seuils = precision_recall_curve(y_test, y_pred_proba)

    precisions = precisions[:-1]

    rappels = rappels[:-1]

    condition = rappels + precisions > 0

    precisions = precisions[condition]

    rappels = rappels[condition]

    seuils = seuils[condition]

    f1_scores = 2 * rappels * precisions/(rappels + precisions)

    val_seuil = seuils[np.argmax(f1_scores)]

    print(classification_report(y_test, y_pred_proba >= val_seuil))

    print("Matrice de confusion du test set:")

    dtf_conf = pd.DataFrame({"cible" : list(y_test), "prev": y_pred_proba >= val_seuil})

    un_moins_spec, sens, seuil = roc_curve(y_test, y_pred_proba)

   

    if auctrain > 0.5:

        print(classification_report(y_test, y_pred_proba >= val_seuil))

        print("Matrice de confusion du test set:")

        dtf_conf = pd.DataFrame(

            {"cible": list(y_test), "prev": y_pred_proba >= val_seuil})

        un_moins_spec, sens, seuil = roc_curve(y_test, y_pred_proba)

        print('OK1')

    cm = confusion_matrix(y_test, y_pred_proba >= val_seuil)

    # Afficher la matrice de confusion

    print("Matrice de confusion du test set:")

    print(cm)

   

    if (cm[0,0]==0) and (cm[1,0]==0):

        print('OK2')

        dff.at[Param,'Variable prédite'] = 'OEDEME'

        dff.at[Param,'Variable explicative'] = Param

        dff.at[Param,'Test U1 de Mann-Whitney sur la totalité des données'] = round(float(U1), 2)

        dff.at[Param,'p'] = float(p)

        dff.at[Param,'Moy Path'] = round(float(MoyKC), 2)

        dff.at[Param,'Moy T'] = round(float(MoyT), 2)

        dff.at[Param,'Med Path'] = round(float(MedKC),2)

        dff.at[Param,'Med T'] = round(float(MedT), 2)

        dff.at[Param,'SD Path'] = round(float(SDKC), 2)

        dff.at[Param,'SD T'] = round(float(SDT), 2)   

        dff.at[Param,'Stats Path train'] = str(StatsPathtrain)

        dff.at[Param,'Stats T train'] = str(StatsTtrain)

        dff.at[Param, 'p train'] = float(ptrain)

    elif auc > 0.5:

        print(pd.crosstab(dtf_conf.cible, dtf_conf.prev, rownames=['Réel'], colnames=['Prédit']))

        VN = pd.crosstab(dtf_conf.cible, dtf_conf.prev, rownames=['Réel'], colnames=['Prédit']).iloc[0,0]

        FP = pd.crosstab(dtf_conf.cible, dtf_conf.prev, rownames=['Réel'], colnames=['Prédit']).iloc[0,1]

        FN = pd.crosstab(dtf_conf.cible, dtf_conf.prev, rownames=['Réel'], colnames=['Prédit']).iloc[1,0]

        VP = pd.crosstab(dtf_conf.cible, dtf_conf.prev, rownames=['Réel'], colnames=['Prédit']).iloc[1,1]

        Se = float((VP/(VP+FN))*100)

        Sp = float((VN/(VN+FP))*100)

        VPP = float((VP/(VP+FP))*100)

        VPN = float((VN/(VN+FN))*100)

        print('Se=', Se, 'Sp=', Sp, 'VPP=', VPP, 'VPN=', VPN)

        print("Meilleur seuil: ", val_seuil)

        print("Meilleur F1-Score: ", np.max(f1_scores))

        recall_0 = 1 - un_moins_spec[sum(seuil>val_seuil)]

        print("Sp(Recall 0, %): ", float(recall_0*100))

        recall_1 = sens[sum(seuil>val_seuil)]

        print("Se (Recall 1, %): ", float(recall_1*100))

        youden = recall_0 + recall_1 - 1

        print('Indice de Youden: ', youden)

        # Créer la courbe ROC du test set

        Threshold=str(val_seuil)

        plt.plot(fpr,tpr,label=str(Param)+" AUC (test set)="+str(auc))

        plt.ylabel("True Positive Rate")

        plt.xlabel("False Positive Rate")

        plt.plot([0,1], [0,1], color = 'navy', linestyle = '--')

        plt.plot(1 - recall_0, recall_1,'ro',label='Threshold='+Threshold); plt.legend();

        plt.legend(loc=4)

        plt.show()

        dff.at[Param,'Variable prédite'] = 'OEDEME'

        dff.at[Param,'Variable explicative'] = Param

        dff.at[Param,'Test U1 de Mann-Whitney sur la totalité des données'] = round(float(U1), 2)

        dff.at[Param,'p'] = float(p)

        dff.at[Param,'Moy Path'] = round(float(MoyKC), 2)

        dff.at[Param,'Moy T'] = round(float(MoyT), 2)

        dff.at[Param,'Med Path'] = round(float(MedKC),2)

        dff.at[Param,'Med T'] = round(float(MedT), 2)

        dff.at[Param,'SD Path'] = round(float(SDKC), 2)

        dff.at[Param,'SD T'] = round(float(SDT), 2)   

        dff.at[Param,'Stats Path train'] = str(StatsPathtrain)

        dff.at[Param,'Stats T train'] = str(StatsTtrain)

        dff.at[Param, 'p train'] = float(ptrain)

        dff.at[Param,'Coefficient'] = float(modele_regLog.coef_[0])

        dff.at[Param,'Constante du modèle'] = float(modele_regLog.intercept_)

        dff.at[Param,'Précision du modèle (VPP, %)'] = round(float(precision*100), 1)

        dff.at[Param,'AUC (courbe ROC, train set)'] = round(float(auctrain), 3)

        dff.at[Param,'Seuil de la variable'] = round(float(seuil_proba_50), 2)

        dff.at[Param,'AUC (courbe ROC, test set)'] = round(float(auc), 3)

        dff.at[Param,'Meilleur seuil'] = float(seuil_f1_max)

        dff.at[Param,'Meilleur F1-Score'] = round(float(np.max(f1_scores)), 2)

        dff.at[Param,'Sp (%)'] = round(float(recall_0*100), 1)

        dff.at[Param,'Se (%)'] = round(float(recall_1*100), 1)

        dff.at[Param,'VPP (%)'] = round(VPP, 1)

        dff.at[Param,'VPN (%)'] = round(VPN, 1)

        dff.at[Param,'Indice de Youden'] = round(float(youden), 2)

        dff.at[Param,'VN'] = VN

        dff.at[Param,'VP'] = VP

        dff.at[Param,'FN'] = FN

        dff.at[Param,'FP'] = FP

        dff.at[Param,'Exactitude'] = round(((VN + VP) / (VN + VP + FN + FP)) * 100, 1)

    else:

        dff.at[Param,'Variable prédite'] = 'OEDEME'

        dff.at[Param,'Variable explicative'] = Param

        dff.at[Param,'Test U1 de Mann-Whitney sur la totalité des données'] = round(float(U1), 2)

        dff.at[Param,'p'] = float(p)

        dff.at[Param,'Moy Path'] = round(float(MoyKC), 2)

        dff.at[Param,'Moy T'] = round(float(MoyT), 2)

        dff.at[Param,'Med Path'] = round(float(MedKC),2)

        dff.at[Param,'Med T'] = round(float(MedT), 2)

        dff.at[Param,'SD Path'] = round(float(SDKC), 2)

        dff.at[Param,'SD T'] = round(float(SDT), 2)   

        dff.at[Param,'Stats Path train'] = str(StatsPathtrain)

        dff.at[Param,'Stats T train'] = str(StatsTtrain)

        dff.at[Param,'Coefficient'] = float(modele_regLog.coef_[0])

        dff.at[Param,'Constante du modèle'] = float(modele_regLog.intercept_)

        dff.at[Param,'Précision du modèle (VPP, %)'] = float(precision*100)

        dff.at[Param,'AUC (courbe ROC, train set)'] = round(float(auctrain), 3)

        dff.at[Param,'AUC (courbe ROC, test set)'] = round(float(auc), 3)      

 

dff = dff.sort_values('AUC (courbe ROC, test set)', ascending=False)

print('\n')

 

#Modèle avec toutes les variables

print('Modèle avec toutes les variables :\n')

x = df[['CBI', 'SSI', 'Pachy', 'DA Ratio', 'ARTh', 'IR', 'Radius', 'DA',

        'V2', 'L2', 'IOPnct', 'L1', 'PD', 'bIOP', 'V1']]

# Fractionner le dataset en un set d’apprentissage (70%) et un set d’évaluation (30%)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in split.split(df, df['OEDEME']):

    train_set = df.loc[train_index]

    test_set = df.loc[test_index]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)

print('Dataframe séparé en un train set et un test set')

dft = df.loc[train_index]

dfs = df.loc[test_index]

print('Train set:', '\n', dft['OEDEME'].value_counts(normalize = True))

print('Test set:', '\n', dfs['OEDEME'].value_counts(normalize = True))

 

#instanciation du modèle

modele_regLog = LogisticRegression(penalty=None,solver='newton-cg')

#training

modele_regLog.fit(x_train,y_train)

#affichage des coefficients et de la constante

print(pd.DataFrame({"var":x_train.columns,"coef":modele_regLog.coef_[0]}))

print('Constante du modèle: ', modele_regLog.intercept_)

#précision du modèle (VPP)

precision = modele_regLog.score(x_test,y_test)

print('Précision du modèle dans le test set (Valeur Prédictive Positive, %): ', precision*100)

print('\n')

 

# Définir les metrics

y_pred_proba = modele_regLog.predict_proba(x_train)[::,1]

fpr, tpr,_=metrics.roc_curve(y_train, y_pred_proba)

auctrain = metrics.roc_auc_score(y_train, y_pred_proba)

print('AUC (aire sous la courbe ROC du train set): ',str(auctrain))

 

# Créer la courbe ROC du train set

plt.plot(fpr,tpr,label= "All parameters AUC (train set)="+str(auctrain))

plt.ylabel("True Positive Rate")

plt.xlabel("False Positive Rate")

plt.plot([0,1], [0,1], color = 'navy', linestyle = '--')

plt.legend(loc=4)

plt.show()

print('\n')

 

# Définir les metrics

y_pred_proba = modele_regLog.predict_proba(x_test)[::,1]

fpr, tpr,_=metrics.roc_curve(y_test, y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

print('AUC (aire sous la courbe ROC du test set): ',str(auc))

print('\n')

precisions, rappels, seuils = precision_recall_curve(y_test, y_pred_proba)

precisions = precisions[:-1]

rappels = rappels[:-1]

condition = rappels + precisions > 0

precisions = precisions[condition]

rappels = rappels[condition]

seuils = seuils[condition]

f1_scores = 2 * rappels * precisions/(rappels + precisions)

val_seuil = seuils[np.argmax(f1_scores)]

print(classification_report(y_test, y_pred_proba >= val_seuil))

print("Matrice de confusion du test set:")

dtf_conf = pd.DataFrame({"cible" : list(y_test), "prev": y_pred_proba >= val_seuil})

un_moins_spec, sens, seuil = roc_curve(y_test, y_pred_proba)

 

cm = confusion_matrix(y_test, y_pred_proba >= val_seuil)

# Afficher la matrice de confusion

print("Matrice de confusion:")

print(cm)

 

if (cm[0,0]==0) and (cm[1,0]==0):

    print('OK2')

    dff.at['AllParameters','Variable prédite'] = 'OEDEME'

    dff.at['AllParameters','Variable explicative'] = Param

elif auc > 0.5:

    print(pd.crosstab(dtf_conf.cible, dtf_conf.prev, rownames=['Réel'], colnames=['Prédit']))

    VN = pd.crosstab(dtf_conf.cible, dtf_conf.prev, rownames=['Réel'], colnames=['Prédit']).iloc[0,0]

    FP = pd.crosstab(dtf_conf.cible, dtf_conf.prev, rownames=['Réel'], colnames=['Prédit']).iloc[0,1]

    FN = pd.crosstab(dtf_conf.cible, dtf_conf.prev, rownames=['Réel'], colnames=['Prédit']).iloc[1,0]

    VP = pd.crosstab(dtf_conf.cible, dtf_conf.prev, rownames=['Réel'], colnames=['Prédit']).iloc[1,1]

    Se = float((VP/(VP+FN))*100)

    Sp = float((VN/(VN+FP))*100)

    VPP = float((VP/(VP+FP))*100)

    VPN = float((VN/(VN+FN))*100)

    print('Se=', Se, 'Sp=', Sp, 'VPP=', VPP, 'VPN=', VPN)

    print("Meilleur seuil: ", val_seuil)

    print("Meilleur F1-Score: ", np.max(f1_scores))

    recall_0 = 1 - un_moins_spec[sum(seuil>val_seuil)]

    print("Sp(Recall 0, %): ", float(recall_0*100))

    recall_1 = sens[sum(seuil>val_seuil)]

    print("Se (Recall 1, %): ", float(recall_1*100))

    youden = recall_0 + recall_1 - 1

    print('Indice de Youden: ', youden)

    # Créer la courbe ROC du test set

    Threshold=str(val_seuil)

    plt.plot(fpr,tpr,label="All parameters AUC (test set)="+str(auc))

    plt.ylabel("True Positive Rate")

    plt.xlabel("False Positive Rate")

    plt.plot([0,1], [0,1], color = 'navy', linestyle = '--')

    plt.plot(1 - recall_0, recall_1,'ro',label='Threshold='+Threshold); plt.legend();

    plt.legend(loc=4)

    plt.show()

    dff.at['AllParameters','Variable prédite'] = 'OEDEME'

    dff.at['AllParameters','Variable explicative'] = 'AllParameters'

    dff.at['AllParameters','Constante du modèle'] = float(modele_regLog.intercept_)

    dff.at['AllParameters','Précision du modèle (VPP, %)'] = round(float(precision*100), 1)

    dff.at['AllParameters','AUC (courbe ROC, train set)'] = round(float(auctrain), 3)

    dff.at['AllParameters','AUC (courbe ROC, test set)'] = round(float(auc), 3)

    dff.at['AllParameters','Meilleur F1-Score'] = round(float(np.max(f1_scores)), 2)

    dff.at['AllParameters','Sp (%)'] = round(float(recall_0*100), 1)

    dff.at['AllParameters','Se (%)'] = round(float(recall_1*100), 1)

    dff.at['AllParameters','VPP (%)'] = round(VPP, 1)

    dff.at['AllParameters','VPN (%)'] = round(VPN, 1)

    dff.at['AllParameters','Indice de Youden'] = round(float(youden), 2)

    dff.at['AllParameters','VN'] = VN

    dff.at['AllParameters','VP'] = VP

    dff.at['AllParameters','FN'] = FN

    dff.at['AllParameters','FP'] = FP

    dff.at['AllParameters','Exactitude']= round(((VN + VP) / (VN + VP + FN + FP)) * 100, 1)

else:

    dff.at['AllParameters','Variable prédite'] = 'OEDEME'

    dff.at['AllParameters','Variable explicative'] = 'AllParameters'

    dff.at['AllParameters','Constante du modèle'] = float(modele_regLog.intercept_)

    dff.at['AllParameters','Précision du modèle (VPP, %)'] = round(float(precision*100), 1)

    dff.at['AllParameters','AUC (courbe ROC, train set)'] = round(float(auctrain), 3)

    dff.at['AllParameters','AUC (courbe ROC, test set)'] = round(float(auc), 3)       

dff = dff.sort_values('AUC (courbe ROC, test set)', ascending=False)

 

# Export des résultats vers Excel

#dff.to_excel(

 #   "Courbes ROC stratifiées Oedème test.xlsx",

 #   sheet_name="ROC",

#)

 

