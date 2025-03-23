from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

# from django.contrib.auth.decorators import login_required
import pandas as pd
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
import scanpy as sc
import numpy as np
from matplotlib import pyplot as plt
import hdbscan
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import StandardScaler
import math
import io
import base64
from django.db.models import Q
from collections import defaultdict
from django.contrib.auth import authenticate

# gene_result.txt, genes_ncbi_proteincoding.py, go-basic.obo

import matplotlib
import re, json, shap
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error

matplotlib.use("agg")
import plotly.graph_objects as go

import requests
from bs4 import BeautifulSoup
from .constants import BUILT_IN_LABELS, NUMBER_CPU_LIMITS, ALLOW_UPLOAD, SHAP_PLOT_DATASET
from .utils import (
    zip_for_vis,
    fromPdtoSangkey,
    GeneID2SymID,
    usrCheck,
    getExpression,
    getMeta,
    generate_jwt_token,
    auth_required,
    getColumnFields,
    expression_clinic_split,
    map_normalized_keys,
    MLparamSetting,
    shared_ml_dml_cf
)

from .models import MetaFileColumn, UploadedFile, SharedFile, Msigdb
from django.db import transaction
from sklearn.model_selection import train_test_split
from IMID.tasks import (
    runIntegrate,
    runDgea,
    runClustering,
    runFeRed,
    runICA,
    runPCA,
    runTopFun,
    calculate_imf_scores,
    run_ora_task,
    run_oraPlot,
    run_Rf,
    run_XGBoost,
    run_LightGBM,
    run_shap,
    run_MLpreprocess,
    run_DMLreport,
    run_DMLreportDis,
    run_CFreport1,
    run_CFreport2,
    run_GB
)

from econml.dml import CausalForestDML

def restLogin(request):
    if request.method != "POST":
        return HttpResponse("Method not allowed.", status=405)
    body = request.body.decode("utf-8")
    try:
        data = json.loads(body)
    except:
        return JsonResponse({"error": "Invalid credentials"}, status=400)
    username = data["username"]
    password = data["password"]
    user = authenticate(username=username, password=password)
    if user is not None:
        token = generate_jwt_token(user)
        return JsonResponse({"token": token})
    return JsonResponse({"error": "Invalid credentials"}, status=400)


@auth_required
def index(request):
    return render(
        request,
        "index.html",
    )


"""
This page will show different shared data to different users within different user groups. If the user is defined in a group,
the programme will filter his corresponding group. Otherwise it will return all shared data.
"""


@auth_required
def tab(request):
    context = {}
    if request.user.groups.exists():
        context["cohorts"] = list(
            SharedFile.objects.filter(
                type1="expression", groups__in=request.user.groups.all()
            )
            .all()
            .values_list("cohort", "label")
        )
    else:
        context["cohorts"] = list(
            SharedFile.objects.filter(type1="expression")
            .all()
            .values_list("cohort", "label")
        )
    return render(request, "tab1.html", {"root": context})
      
@auth_required
def compICA(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    preload = getColumnFields("0", request, usr)
    preload = [i[0] for i in preload if i[1]=="1"]#Label
    return render(request, "compTypeICA.html", { "root":{"labels":preload, "cID": usr.cID}})

@auth_required        
def compICAcohort(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    if usr.ica_cohort==():
        return HttpResponse('Please Run ICA based cohort first!', status=400)
    return JsonResponse({"text": usr.ica_cohort[:3]})   
        
@auth_required        
def compICAall(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    if usr.ica_cohort==():
        ica_cohort=[]
        X11 = usr.metagenes[0].columns.to_list()
        X22 = ''
    else:
        X11, X22 =usr.metagenes[0].columns.to_list(), usr.metagenes[3].columns.to_list()
        ica_cohort = usr.ica_cohort[:3]
    #temp1=pd.concat([X11,X22],axis=1)#0
    return JsonResponse({"ica_cohort": ica_cohort,"ica_cohort_metagenes":(X11, X22)})  
  
@auth_required
def compORA(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    preload = getColumnFields("0", request, usr)
    preload = [i[0] for i in preload if i[1]=="1"]
    return render(request, "compTypeORA.html", { "root":{"labels":preload, "cID": usr.cID}})



"""
This program provides entrance for user to upload/get Expression files. One user can upload multiple expression files at a time.
'ID_REF' should be one of the colnames in each Expression file.
User uses UploadedFile to store their uploaded expression data, the column type1 is set to be 'exp' to represent the stored expression file.
"""


@auth_required
def opExpression(request):
    if request.method == "POST":
        if ALLOW_UPLOAD is False:
            return HttpResponse("Not Allowed user upload", status=400)
        cID = request.POST.get("cID", None)
        if cID is None:
            return HttpResponse("cID not provided.", status=400)
        UploadedFile.objects.filter(user=request.user, type1="exp", cID=cID).delete()
        files = request.FILES.getlist("files[]", None)
        new_exp_files = []

        for f in files:
            temp_file = UploadedFile(user=request.user, cID=cID, type1="exp", file=f)
            new_exp_files.append(temp_file)
        UploadedFile.objects.bulk_create(new_exp_files)
        return getExpression(request, cID)
    elif request.method == "GET":
        cID = request.GET.get("cID", None)
        if cID is None:
            return HttpResponse("cID not provided.", status=400)
        return getExpression(request, cID, 0)


"""
This program provides entrance for user to upload/get Meta files. One user can upload only one meta file at a time.
'ID_REF','LABEL' should be one of the colnames in each Meta file.
User uses UploadedFile to store their uploaded expression data, the column type1 is set to be 'cli' to represent the stored clinical data.
"""


@auth_required
def opMeta(request):
    if request.method == "POST":
        if ALLOW_UPLOAD is False:
            return HttpResponse("Not Allowed user upload", status=400)
        cID = request.POST.get("cID", None)
        if cID is None:
            return HttpResponse("cID not provided.", status=400)
        files = request.FILES.getlist("meta", None)
        if files is None:
            return HttpResponse("Upload the meta file is required", status=405)
        files = files[0]
        UploadedFile.objects.filter(user=request.user, type1="cli", cID=cID).delete()
        UploadedFile.objects.create(user=request.user, cID=cID, type1="cli", file=files)
        return getMeta(request, cID)
    elif request.method == "GET":
        cID = request.GET.get("cID", None)
        if cID is None:
            return HttpResponse("cID not provided.", status=400)
        return getMeta(request, cID, 0)


"""
This is for data integration based on selected options. The files comes from 1. user uploaded files. 2. Built-in Data.
in_ta and in_ta1 are used for expression and meta data list separately
The processing logic is:
First, using user uploaded meta file as the base then inner join with selected built-in meta files to get the shared clinic features. =>temp0
Then join the expression files=>dfs1 with consideration of log2, batch effect
Third, join temp0 and dfs1;
"""


@auth_required
def edaIntegrate(request):
    checkRes = usrCheck(request, 0)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    cID = request.GET.get("cID", None)

    corrected = request.GET.get("correct", "Combat")
    log2 = request.GET.get("log2", "No")
    fr = request.GET.get("fr", "TSNE")
    integrate = [i.strip() for i in request.GET.get("integrate", "").split(",")]
    if "" in integrate:
        integrate.remove("")

    try:
        usr.log2 = log2
        if usr.save() is False:
            return HttpResponse("Can't save user record", status=500)
        result = runIntegrate.apply_async(
            (request, integrate, cID, log2, corrected, usr, fr), serializer="pickle"
        ).get()
    except Exception as e:
        return HttpResponse(str(e), status=500)
    return HttpResponse("Operation successful.", status=200)


@auth_required
def eda(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    adata = usr.getAnndata()
    targetLabel = request.GET.get("label", "batch2")
    color2 = adata.obs[targetLabel]  # temp.LABEL, temp.FileName
    X2D = usr.getFRData()
    traces = zip_for_vis(
        X2D.tolist(), adata.obs["batch1"], adata.obs_names
    )  # temp.FileName, temp.obs
    traces1 = zip_for_vis(X2D.tolist(), color2, adata.obs_names)
    labels = list(
        MetaFileColumn.objects.filter(user=request.user, cID=usr.cID, label="1")
        .all()
        .values_list("colName", flat=True)
    )
    if "LABEL" in labels:
        labels.remove("LABEL")
        labels = ["LABEL"] + labels
    context = {
        "dfs1": traces,
        "dfs2": traces1,
        "labels": labels,
        "method": usr.redMethod,
    }
    return JsonResponse(context)


@auth_required
def dgea(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    targetLabel = request.GET.get("label", "batch2")
    adata = usr.getAnndata()

    clusters = request.GET.get("clusters", "default")
    n_genes = request.GET.get("topN", 4)

    try:
        result = runDgea.apply_async(
            (clusters, adata, targetLabel, n_genes), serializer="pickle"
        ).get()
    except Exception as e:
        return HttpResponse(str(e), status=500)
    if type(result) is list:
        return JsonResponse(result, safe=False)
    else:
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=topGenes.csv"
        result.to_csv(path_or_buf=response)
        return response


@auth_required
def clustering(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    cluster = request.GET.get("cluster", "LEIDEN")
    param = request.GET.get("param", None)
    if param is None:
        return HttpResponse("Param is illegal!", status=400)
    X2D = usr.getFRData()
    if X2D is None:
        return HttpResponse("Please run feature reduction first.", status=400)
    X2D = X2D.tolist()
    adata = usr.getAnndata()
    try:
        result = runClustering.apply_async(
            (cluster, adata, X2D, usr, param), serializer="pickle"
        ).get()
    except Exception as e:
        return HttpResponse(str(e), status=400)
    return JsonResponse(result)


@auth_required
def clusteringAdvanced(request):
    clientID = request.GET.get("cID", None)
    if clientID is None:
        return HttpResponse("clientID is Required", status=400)
    if request.method == "GET" and "cluster" not in request.GET:
        return render(request, "clustering_advance.html", {"cID": clientID})
    else:
        checkRes = usrCheck(request)
        if checkRes["status"] == 0:
            return HttpResponse(checkRes["message"], status=400)
        else:
            usr = checkRes["usrData"]
        cluster = request.GET.get("cluster", "LEIDEN")
        minValue = float(request.GET.get("min", "0"))
        maxValue = float(request.GET.get("max", "1"))
        level = int(request.GET.get("level", 3))
        adata = usr.getAnndata().copy()
        df = usr.getCorrectedCSV()[["LABEL"]]
        if level > 10 or level <= 1:
            return HttpResponse("Error for the input", status=400)
        if cluster == "LEIDEN":
            if minValue <= 0:
                minValue = 0
            if maxValue >= 2:
                maxValue = 2
            for i, parami in enumerate(np.linspace(minValue, maxValue, level)):
                with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
                    sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
                    sc.tl.leiden(adata, resolution=float(parami))
                df["level" + str(i + 1)] = [
                    "level" + str(i + 1) + "_" + str(j) for j in adata.obs["leiden"]
                ]
        elif cluster == "HDBSCAN":
            if minValue <= 5:
                minValue = 5
            if maxValue >= 100:
                maxValue = 100
            for i, parami in enumerate(np.linspace(minValue, maxValue, level)):
                df["level" + str(i + 1)] = [
                    "level" + str(i + 1) + "_" + str(j)
                    for j in hdbscan.HDBSCAN(min_cluster_size=int(parami)).fit_predict(
                        adata.obsm["X_pca"]
                    )
                ]

        result = fromPdtoSangkey(df)
        return JsonResponse(result)


@auth_required
def advancedSearch(request):
    # username = request.user.username
    name = request.GET.get("name", None)
    clientID = request.GET.get("cID", None)
    if clientID is None:
        return HttpResponse("clientID is Required", status=400)
    res = {}
    if name is None:
        return render(request, "advancedSearch.html", {"clientID": clientID})
    name = name.replace(" ", "")
    res["name"] = name
    url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=" + name
    page = requests.get(url)
    p = page.content.decode("utf-8").replace("\n", "")
    if "GEO accession display tool" in p:
        return HttpResponse("Could not find the cohort!", status=404)
    m = re.search(
        '(?<=\<tr valign="top"\>\<td nowrap\>Summary\</td\>\<td style="text-align: justify"\>)([\w\s,-.!"\(\):%]+)',
        p,
    )
    if m is not None:
        res["summary"] = m.group(1)
    m = re.search(
        '(?<=\<tr valign="top"\>\<td nowrap\>Overall design\</td\>\<td style="text-align: justify"\>)([\w\s,-.!"\(\):%]+)',
        p,
    )
    if m is not None:
        res["overAllDesign"] = m.group(1)
    m = re.search(
        '(\<table cellpadding="2" cellspacing="2" width="600"\>\<tr bgcolor="#eeeeee" valign="top"\>\<td align="middle" bgcolor="#CCCCCC"\>\<strong\>Supplementary file\</strong\>\</td\>)(.+)(\</tr\>\<tr\>\<td class="message"\>[\w\s]+\</td\>\</tr\>\</table\>)',
        p,
    )
    if m is not None:
        soup = BeautifulSoup(m.group(0), "html.parser")
        ftp_tags = soup.find_all("a", string="(ftp)")
        for ftp in ftp_tags:
            ftp.decompose()  # remove all ftp nodes
        custom_tags = soup.find_all("a", string="(custom)")
        for cus in custom_tags:
            cus.decompose()
        td_tags = soup.find_all("td", class_="message")
        for td in td_tags:
            td.decompose()
        td_tags = soup.find_all("td")
        for td in td_tags:
            if "javascript:" in str(td):
                td.decompose()
        res["data"] = str(soup).replace(
            "/geo/download/", "https://www.ncbi.nlm.nih.gov/geo/download/"
        )
    if "TXT" in p:
        res["txt"] = 1
    else:
        res["txt"] = 0
    return JsonResponse(res)


@auth_required
def goenrich(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]   
    cohort = request.GET.get("cohort", None)
    metagene = request.GET.get("metagene", None)
    if metagene is None:
        return HttpResponse("Illegal Metagene Name.", status=400)
    if cohort is None and (len(usr.metagenes)!=1 or len(usr.metageneCompose)==0):
        return HttpResponse("Please run ICA first.", status=400)
    if cohort is not None and len(usr.metagenes)!=4:
        return HttpResponse("Please run ICA-cohort first.", status=400)
    try:
        if cohort is None or cohort=='A':
            result = runTopFun.apply_async(
                (usr.metageneCompose[0], metagene), serializer="pickle"
            ).get()
        else:
            result = runTopFun.apply_async(
                (usr.metageneCompose[1], metagene), serializer="pickle"
            ).get()
    except Exception as e:
        return HttpResponse(str(e), status=400)
    return JsonResponse(result, safe=False)


@auth_required
def downloadICA(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    typ = request.GET.get("type", 'ica')
    if typ=='ica':
        if len(usr.metagenes)!=1:
            return HttpResponse("Please run ICA first.", status=400)
        result_df = usr.metagenes[0]
    else:
        if len(usr.metagenes)!=4:
            return HttpResponse("Please run ICA-cohort first.", status=400)
        X11, X12, X21, X22 = usr.metagenes
        temp1 = pd.concat([X11,X22], axis=1)
        temp2 = pd.concat([X12,X21], axis=1)
        result_df = pd.concat([temp1, temp2], axis=0, join='inner')
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=ica.csv"
    result_df.to_csv(path_or_buf=response)
    return response


@auth_required
def downloadCorrected(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    result_df = usr.getCorrectedCSV()
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=corrected.csv"
    result_df.to_csv(path_or_buf=response)
    return response


@auth_required
def downloadCorrectedCluster(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    result_df = usr.getCorrectedClusterCSV()
    if result_df is None:
        return HttpResponse("Please do clustering first.", status=400)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=correctedCluster.csv"
    result_df.to_csv(path_or_buf=response)
    return response


@auth_required
def candiGenes(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    method = request.GET.get("method", None)
    maxGene = 12
    if method is None or method == "pca":
        adata = usr.getAnndata()
        n_pcs = 3
        pcs_loadings = pd.DataFrame(adata.varm["PCs"][:, :n_pcs], index=adata.var_names)
        pcs_loadings.dropna(inplace=True)
        result = []
        for i in pcs_loadings.columns:
            result.extend(pcs_loadings.nlargest(2, columns=i).index.tolist())
            result.extend(pcs_loadings.nsmallest(2, columns=i).index.tolist())
        return JsonResponse(result, safe=False)
    else:
        markers = usr.getMarkers(method)
        if markers is None:
            return HttpResponse("Please run clustering method first.", status=400)
        try:
            clusters = set(markers.group)
        except Exception as e:
            return HttpResponse(
                "There is no different values for the label you chose.", status=400
            )
        number = math.ceil(maxGene / len(clusters))
        markers = markers.query("scores > 0")
        result = (
            markers.groupby("group")
            .apply(lambda x: x.nsmallest(number, "pvals_adj"))
            .names.tolist()
        )
        return JsonResponse(result, safe=False)



@auth_required
def GeneLookup(request):
    geneList = request.GET.get("geneList", None)
    if geneList is None:
        return HttpResponse("Gene List is required", status=400)
    try:
        geneList = geneList.split(",")
        geneList = [i for i in geneList if i != "None"]
    except:
        return HttpResponse("Gene List is illegal", status=400)
    result = GeneID2SymID(geneList)
    if result is None:
        return HttpResponse("Gene List contains no Ensembl gene IDs (prefix \"ENSG\") to convert", status=400)
    return JsonResponse(result, safe=False)


@auth_required
def checkUser(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        return HttpResponse("User exists.", status=200)


@auth_required
def meta_columns(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    cID = request.GET.get("cID", None)
    if request.method == "GET":
        numeric = request.GET.get("numeric", None)
        param = request.GET.get("param",1)
        if numeric is None:
            result = [
                [i[0], i[1], i[2]]
                for i in MetaFileColumn.objects.filter(user=request.user, cID=usr.cID)
                .all()
                .values_list("colName", "label", "numeric")
            ]
            if param=="2":
                created =[i[0].split('__crted')[0] for i in result if 'crted' in i[0]]
                result = [i for i in result if i[0] not in created]
        else:
            result = [
                [i[0], i[1], i[2]]
                for i in MetaFileColumn.objects.filter(user=request.user, cID=usr.cID, numeric="0")
                .all().values_list("colName", "label", "numeric")
            ]
            if param =="2":
                created =[i[0].split('__crted')[0] for i in result if 'crted' in i[0]]
                result2 = [
                    [i[0], i[1], i[2]]
                    for i in MetaFileColumn.objects.filter(
                        user=request.user, cID=usr.cID, numeric=numeric
                    )
                    .all()
                    .values_list("colName", "label", "numeric")
                ]       
                result = [i for i in result2 if i[0] not in created]
        return JsonResponse(result, safe=False)
    elif request.method == "PUT":
        labels = request.GET.get("labels", None)
        fr = request.GET.get("fr", "TSNE")
        if labels is None:
            return HttpResponse("Labels illegal.", status=400)
        labels = [i.strip() for i in labels.split(",")]
        error_labels = BUILT_IN_LABELS.intersection(set(labels))
        if len(error_labels) != 0:
            return HttpResponse(
                "Labels creating Problem, can't use retained words as an label:"
                + str(error_labels),
                status=400,
            )
        try:
            with transaction.atomic():
                df = usr.getIntegrationData().copy()
                crtedDic = defaultdict(list)
                for i in df.columns:
                    if "__crted" in i:
                        crtedDic[i.split("__crted")[0] + "__crted"].append(i)
                if labels != [""]:
                    labels_t = labels.copy()
                    labels1 = set(
                        [
                            i.split("__crted")[0] + "__crted"
                            for i in labels
                            if "__crted" in i
                        ]
                    )
                    labels2 = [i for i in labels if "__crted" not in i]
                    labels = set()
                    for i in crtedDic:
                        if i in labels1:
                            labels.update(crtedDic[i])  # add age__crted1-N
                            labels.add(i.split("__crted")[0])  # add age
                        else:
                            labels.update(crtedDic[i])  # add agg_crted1-N
                    labels.update(labels2)
                    df1 = df.drop(labels, axis=1, inplace=False)
                    usr.setAnndata(df1)
                    adata = usr.getAnndata()
                    for label in labels_t:
                        if not np.issubdtype(df[label].dtype, np.number):
                            adata.obs[label] = df[label]
                        else:
                            raise Exception(
                                "Can't auto convert numerical value for label."
                            )

                    MetaFileColumn.objects.exclude(colName="LABEL").exclude(
                        user=request.user, cID=cID, colName__in=labels_t
                    ).update(label="0")
                    MetaFileColumn.objects.filter(
                        user=request.user, cID=cID, colName__in=labels_t
                    ).update(label="1")
                else:
                    usr.setIntegrationData(df)
                    MetaFileColumn.objects.exclude(colName="LABEL").filter(
                        user=request.user, cID=cID
                    ).update(label="0")
                X2D = runFeRed.apply_async((fr, usr), serializer="pickle").get()
                usr.setFRData(X2D)
        except Exception as e:
            return HttpResponse("Labels creating Problem. " + str(e), status=400)
        finally:
            if usr.save() is False:
                return HttpResponse("Can't save user record", status=500)
        return HttpResponse("Labels updated successfully.", status=200)
    elif request.method == "POST":
        post_data = json.loads(request.body)
        colName = post_data.get("colName")
        threshold = post_data.get("thredshold")
        threshold_labels = post_data.get("thredshold_labels")
        if (
            colName is None
            or threshold is None
            or threshold_labels is None
            or len(threshold) != len(threshold_labels) - 1
        ):
            return HttpResponse("Illegal Param Input. ", status=400)

        threshold = [float(i) for i in threshold]
        count = MetaFileColumn.objects.filter(
            (Q(colName=colName) | Q(colName__startswith=colName + "__crted"))
            & Q(user=request.user)
        ).count()
        if count == 0:
            HttpResponse(
                "No such colName: " + str(colName) + " in meta file.", status=400
            )
        colName1 = colName + "__crted" + str(count)
        df, adata = usr.getIntegrationData(), usr.getAnndata()
        conditions = [df[colName] < threshold[0]]
        for i in range(len(threshold_labels) - 2):
            conditions.append(
                (df[colName] >= threshold[i]) & (df[colName] < threshold[i + 1])
            )
        conditions.append(df[colName] >= threshold[-1])

        try:
            with transaction.atomic():
                df[colName1] = np.select(conditions, threshold_labels)
                adata.obs[colName1] = df[colName1].copy()
                MetaFileColumn.objects.create(
                    user=request.user, cID=cID, colName=colName1, label="0", numeric="0"
                )
        except Exception as e:
            return HttpResponse("Labels creating Problem. " + str(e), status=400)
        finally:
            if usr.save() is False:
                return HttpResponse("Can't save user record", status=500)
        return HttpResponse("Label created Successfully. ", status=200)

@auth_required
def meta_column_values(request, colName):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    adata = usr.getAnndata()
    df = usr.getIntegrationData()
    if colName.lower() == "Cluster".lower():
        colName = "cluster"
    if request.method == "GET":
        if colName == 'LABEL':
            colName = 'batch2'
        if colName in adata.obs_keys() and not np.issubdtype(
            adata.obs[colName].dtype, np.number
        ):
            temp = list(set(adata.obs[colName]))
            if len(temp) == 1:
                return HttpResponse(
                    "Only 1-type value found in the colName: " + colName, status=400
                )
            elif len(temp) > 30:
                return HttpResponse(
                    "More than 30-type values found in the colName: " + colName,
                    status=400,
                )
            temp.sort()
            return JsonResponse(temp, safe=False)
        if colName in df.columns:
            histogram_trace = go.Histogram(
                x=df[colName],
                histnorm="probability density",  # Set histogram normalization to density
                marker_color="rgba(0, 0, 255, 0.7)",  # Set marker color
            )

            # Configure the layout
            layout = go.Layout(
                title="Density Plot for " + colName,  # Set plot title
                xaxis=dict(title=colName),  # Set x-axis label
                yaxis=dict(title="Density"),  # Set y-axis label
            )

            # Create figure
            fig = go.Figure(data=[histogram_trace], layout=layout)

            return HttpResponse(
                base64.b64encode(fig.to_image(format="svg")).decode("utf-8"),
                content_type="image/svg+xml",
            )
        else:
            return HttpResponse("Can't find the colName: " + colName, status=400)
    if request.method == "DELETE":
        col = MetaFileColumn.objects.filter(
            user=request.user, cID=usr.cID, colName=colName
        ).first()
        if col is None:
            return HttpResponse("No such colName called:" + colName, status=400)
        elif col.label == "1":
            return HttpResponse("Please make {colName} inactive first.", status=400)
        MetaFileColumn.objects.filter(
            user=request.user, cID=usr.cID, colName=colName, label="0"
        ).delete()
        return HttpResponse("Delete {colName} Successfully.", status=200)

from urllib.parse import unquote
@auth_required
def ICAreport(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    method = request.GET.get("method", 'ica')
    imf = request.GET.get("imf","no")
    if method=='ica_cohort':
        imf='no'
    p = request.GET.get('p', '')
    p0 = unquote(request.GET.get('p0', ''))
    p1 = unquote(request.GET.get('p1', ''))
    bridge = request.GET.get('bridge','no')
    if method =='ica_cohort' and (p0=='' or p1=='' or p==''):
        return HttpResponse("Empty p0 or p1 or p.", status=400)
    if p=='LABEL':
        p='batch2'
        
    adata = usr.getAnndata().copy()
    dfe, dfc = expression_clinic_split(adata)
    try:
        dim = int(request.GET.get("dim",7))
    except Exception:
        return HttpResponse("Dim should be an integer", status=500)
    if dim >= dfe.shape[0]:
        return HttpResponse("Dimension is too large!" , status=500)     
    if method=='ica':
        try:
            if imf=='yes':
                imf_score = calculate_imf_scores.apply_async((dfe,),serializer="pickle")
            metageneCompose, metagenes, _, _ = runICA.apply_async((dfe, dim,''),serializer="pickle").get()
        except Exception as e:
            return HttpResponse("ICA Failed:" + str(e), status=500)
        usr.metagenes = [metagenes,]
        usr.metageneCompose = [metageneCompose,]
        if imf=='yes':
            usr.imf = imf_score.get()[0]
        else:
            usr.imf = None
        usr.ica_cohort = ()
        if usr.save() is False:
            return HttpResponse("Can't save user record", status=500)
        if imf=='yes':
            return render(request, "imf_template.html", {"svg_content": imf_score.get()[1], "title":"IMF Score Visualisation", "message":""})
        else:
            return HttpResponse("ICA Successful.", status=200)
    elif method=='ica_cohort':
        usr.imf = None
        if p not in adata.obs_keys():
            return HttpResponse("No such column:" +p, status=500)
        dics = map_normalized_keys([p0,p1],set(adata.obs[p].to_list()))
        if dics[p0] is None or dics[p1] is None:
            return HttpResponse("No such value: "+p0+" / " +p1+" for column:" +p, status=500)
        flag1 = (adata.obs[p]==dics[p0])      
        flag2 = (adata.obs[p]==dics[p1])
        if dim <= sum(flag1) and dim <= sum(flag2):            
            try:
                if bridge=="no":             
                    metageneComposeA, metagenesXX11, metagenesXX12, sICA = runICA.apply_async((dfe, dim, p0[:3], flag1, flag2),serializer="pickle").get()
                    metageneComposeB, metagenesXX21, metagenesXX22, _ = runICA.apply_async((dfe, dim, p1[:3], flag2, flag1),serializer="pickle").get()
                    usr.metagenes = [metagenesXX11, metagenesXX12, metagenesXX21, metagenesXX22]
                    usr.metageneCompose = [metageneComposeA, metageneComposeB]
                else:
                    if len(usr.metagenes)!=4:
                        return HttpResponse("Run ICA-cohort without bridge first. ", status=500)  
                    metageneComposeC, metagenesXX41, metagenesXX42, sICA = runICA.apply_async((dfe, dim, p1[:3], flag2, flag1),serializer="pickle").get()
                    TEMP = usr.ica_cohort[3].transform(dfe[flag2])
                    TEMP = pd.DataFrame(TEMP)
                    TEMP.columns = usr.metagenes[0].columns.copy()
                    TEMP.index = dfe[flag2].index.copy()
                    usr.metagenes = [usr.metagenes[0], TEMP, metagenesXX41, metagenesXX42]
                    usr.metageneCompose = [usr.metageneCompose[0], metageneComposeC]
            except Exception as e:
                return HttpResponse("ICA Failed:" + str(e), status=500)
            usr.ica_cohort = (p, dics[p0], dics[p1], sICA)
            # print(adata.shape, p, dics[p0], dics[p1], metagenesXX11.shape, metagenesXX12.shape)
            # print(metagenesXX11)
        else:
            return HttpResponse("Dimension is too large for cohorts " +str(sum(flag1))+" vs "+str(sum(flag2)), status=500) 
        if usr.save() is False:
            return HttpResponse("Can't save user record", status=500)
        return HttpResponse("ICA Successful.", status=200)
    else:#PCA
        try:
            if imf=='yes':
                imf_score = calculate_imf_scores.apply_async((dfe,),serializer="pickle")
            metagenes = runPCA.apply_async((dfe, dim),serializer="pickle").get()
        except Exception as e:
            return HttpResponse("PCA Failed:" + str(e), status=500)
        usr.metagenes = [metagenes,]
        usr.metageneCompose = []
        if imf=='yes':
            usr.imf = imf_score.get()[0]
        else:
            usr.imf = None
        usr.ica_cohort = ()
        if usr.save() is False:
            return HttpResponse("Can't save user record", status=500)
        if imf=='yes':
            return render(request, "imf_template.html", {"svg_content": imf_score.get()[1], "title":"IMF Score Visualization", "message":""})
        else:
            return HttpResponse("PCA Successful.", status=200)

      

          
@auth_required
def ORAreport(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    db = request.GET.get("msigdb", 'reactome_pathways')
    de = request.GET.get("diff", None)
    group = request.GET.get("group", None)
    top = request.GET.get("top", None)
    if top is None and usr.ora is None:
        return HttpResponse("Run ORA first.", status=500)
    if top is None:
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="ora_data.csv"'
        usr.ora.to_csv(response, index=False)
        return response
    if de is None or group is None:
        return HttpResponse("Not enough parameters", status=500)
    if de=='LABEL':
        de='batch2'
    msigdb = Msigdb.objects.filter(collection=db)
    
    if len(msigdb)==0:
        return HttpResponse("Not found msigdb: "+db, status=500)
    data_list = list(msigdb.values('symbol', 'collection', 'geneset'))
    msigdb1 = pd.DataFrame(data_list)
    msigdb1.columns = ['genesymbol', 'collection', 'geneset']
    msigdb1 = msigdb1[~msigdb1.duplicated(['geneset', 'genesymbol'])]
    
    adata = usr.getAnndata().copy()
    dfe, _ = expression_clinic_split(adata)
    
    adata1 = sc.AnnData(np.zeros(dfe.values.shape), dtype=np.float64)
    adata1.X = dfe.values
    adata1.var_names = dfe.columns.tolist()
    adata1.obs_names = dfe.index.tolist()
    adata1.obs = adata.obs.copy()
    try:
        top = int(top)
        dff, acts=run_ora_task.apply_async((db,adata1,de,msigdb1),serializer="pickle").get()
    except Exception as e:
        return HttpResponse(f"Error during ORA computation: {e}", status=500)
    if usr.save() is False:
        return HttpResponse("Can't save user record", status=500)
    #dff, acts = pd.read_csv('usr_ora.csv'), ''
    usr.ora = (dff,acts)
    if usr.save() is False:
        return HttpResponse("Can't save user record", status=500)
    context = {'root': {"df":dff.head(top).to_dict(orient='records'), "cID": usr.cID, "group":group}}
    return render(request, "ORAreport.html", context)

@auth_required
def ORADE(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    if usr.ora is None:
        return HttpResponse("Run ORA first.", status=500)
    thre = request.GET.get("threshold", 0.05)
    group = request.GET.get("group", "batch2")
    dff, acts = usr.ora
    try:
        source_markers = dff[dff.pvals_adj<float(thre)].groupby('group')['names'].apply(lambda x: list(x)).to_dict()
        image_data = run_oraPlot.apply_async((acts, source_markers, group),serializer="pickle")
    except Exception as e:
        return HttpResponse("Internal error. "+str(e), status=500)
    return JsonResponse({"image": image_data.get()})

@auth_required
def goML(request, mlType, mlMethod):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    #ml/classification/rf/?cID=12345&label=LABEL&class=nonrespondershareORBIT&testRatio=2
    #ml/regression/rf/?cID=12345&label=cage&logY=no&testRatio=2
    # print(len(usr.metagenes))
    try:
        body = json.loads(request.body)
        ml_param = body.get('ml_params', '')
        drop = float(body.get('drop',0.65))
        label = request.GET.get("label", None)
        cla = request.GET.get("class", None)
        testRatio=request.GET.get('testRatio',None)
        logY = request.GET.get('logY',None)
        mlType, param, X, y, testRatio, _ = run_MLpreprocess.apply_async((label, cla, testRatio, logY, usr, mlMethod, mlType, 0, ml_param, drop),serializer="pickle").get()
        # print(mlType)
        # print(param)
        # print(X)
        # print(y)
        # print(testRatio)
        # print(len(usr.metagenes))
    except Exception as e:
        return HttpResponse(str(e), status=500)
    if mlType=='classification':    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, stratify=y, random_state=42)
        # print(X.shape,X_train.shape,X_test.shape)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=42)
    if mlMethod=='rf':
        result=run_Rf.apply_async((mlType, param, X, X_train, y_train, X_test, y_test,testRatio),serializer="pickle").get()
    elif mlMethod=='xgb':
        result=run_XGBoost.apply_async((mlType, param, X, X_train, y_train, X_test, y_test, testRatio),serializer="pickle").get()
    else:
        result=run_LightGBM.apply_async((mlType, param, X, X_train, y_train, X_test, y_test, testRatio),serializer="pickle").get()
    usr.model=[result[1],]
    if usr.save() is False:
        return HttpResponse("Can't save user record", status=500)
    return JsonResponse({"result": result[0]})
        
@auth_required
def goDML(request, mlType, mlMethod,number):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    #dml/classification/rf/1/?cID=12345&label=LABEL&class=nonrespondershareORBIT&testRatio=2
    #dml/regression/rf/1/?cID=12345&label=cage&logY=no&testRatio=2
    try:
        body = json.loads(request.body)
        ml_params = body.get('ml_params', '')
        drop = float(body.get('drop',0.65))
        label = request.GET.get("label", None)
        cla = request.GET.get("class", None)
        testRatio=request.GET.get('testRatio',None)
        logY = request.GET.get('logY',None)
        mlType, param, X, y, testRatio, _ = run_MLpreprocess.apply_async((label, cla, testRatio, logY, usr, mlMethod, mlType, 1, ml_params, drop),serializer="pickle").get()
        X['y'] =y
        temp = X.copy()
        adata = usr.getAnndata().copy()
        if number==1:
            condition = adata.obs[usr.ica_cohort[0]]==usr.ica_cohort[1]
        else:
            condition = adata.obs[usr.ica_cohort[0]]==usr.ica_cohort[2]
        valid_indices = condition[condition].index
        X = temp.loc[valid_indices].drop(columns=['y'])
        y = temp.loc[valid_indices,'y'].values
    except Exception as e:
        return HttpResponse(str(e), status=500)
    if mlType=='classification':    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, stratify=y, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=42)
    if mlMethod=='rf':
        result=run_Rf.apply_async((mlType, param, X, X_train, y_train, X_test, y_test, testRatio),serializer="pickle").get()
    elif mlMethod=='xgb':
        result=run_XGBoost.apply_async((mlType, param, X, X_train, y_train, X_test, y_test, testRatio),serializer="pickle").get()
    else:
        result=run_LightGBM.apply_async((mlType, param, X, X_train, y_train, X_test, y_test, testRatio),serializer="pickle").get()
    if number==1:
        usr.DMLmodel=[result[1],]
    else:
        usr.DMLmodel=[usr.DMLmodel[0],result[1]]
    if usr.save() is False:
        return HttpResponse("Can't save user record", status=500)
    return JsonResponse({"result": result[0]})

@auth_required
def feaImp(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    flip = request.GET.get("flip", 'no')
    if len(usr.model)==0:
        return HttpResponse("Please run ML first.", status=500)
    if len(usr.model)==1:
        image_data, usr.shapCache = run_shap.apply_async((usr.model[0][0], usr.model[0][1],flip),serializer="pickle").get()
    else:
        image_data = ''
    if usr.save() is False:
        return HttpResponse("Can't save user record", status=500)
    return render(request, "shap_importance.html", {"svg_content": image_data, "clientID":usr.cID})

@auth_required
def DMLfeaImp(request, number):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    if len(usr.DMLmodel)<number:
        return HttpResponse("Please run DML first.", status=500)
    if number==1:
        image_data, usr.shapCache = run_shap.apply_async((usr.DMLmodel[0][0], usr.DMLmodel[0][1]),serializer="pickle").get()
    else:
        image_data, usr.shapCache = run_shap.apply_async((usr.DMLmodel[1][0], usr.DMLmodel[1][1]),serializer="pickle").get()
    if usr.save() is False:
        return HttpResponse("Can't save user record", status=500)
    return render(request, "shap_importance.html", {"svg_content": image_data,"clientID":usr.cID})
        
@auth_required
def feaImpData(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    if usr.shapCache is None:
        return HttpResponse("Please run Shap Importance first.", status=500)
    method = request.GET.get("method", 'abs')
    shap_values = usr.shapCache
    if method =='abs':
        shap_importance = np.abs(shap_values.values).mean(axis=0)
    else:
        shap_importance = np.ptp(shap_values.values, axis=0)  # np.ptp computes peak-to-peak (max - min)
        
    feature_names = shap_values.feature_names
    # print(feature_names)
    result_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Importance": shap_importance
    })
    result_df.sort_values(by="SHAP_Importance", ascending=False, inplace=True)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=shap_importance.csv"
    result_df.to_csv(path_or_buf=response, index=False)
    return response
          
@auth_required
def DMLreport(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    if len(usr.DMLmodel)!=2:
        return HttpResponse("Please run 2 DML models first.", status=500)
    image_data, usr.shapImportance, shap_importance_top6 = run_DMLreport.apply_async((usr.DMLmodel[0][0], usr.DMLmodel[0][1], usr.DMLmodel[1][0],usr.DMLmodel[1][1]), serializer="pickle").get()
    if usr.save() is False:
        return HttpResponse("Can't save user record", status=500) 
    return render(request, "DMLshap_importance.html", {"svg_content": image_data, 
        "feature_names": ", ".join(shap_importance_top6),
        "clientID":usr.cID })
        
     
@auth_required
def DMLreportDis(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    param = request.GET.get("param", None)
    if param is None:
        return HttpResponse("No input Feature names", status=404)
    
    metagenes = param.replace("\t", "").replace(" ", "").split(',')
    # Assuming result_final1 contains one tuple with two SHAP importance sorted dictionaries
    if len(usr.shapImportance)!=2:
        return HttpResponse("Please run DML first.", status=500)
    shap_importance_sorted1 = pd.DataFrame(usr.shapImportance[0].values, columns=usr.DMLmodel[0][1].columns).abs().mean(axis=0)
    shap_importance_sorted2 = pd.DataFrame(usr.shapImportance[1].values, columns=usr.DMLmodel[1][1].columns).abs().mean(axis=0)
    
    image_data = run_DMLreportDis.apply_async((metagenes, shap_importance_sorted1, shap_importance_sorted2, usr.ica_cohort[:3]), serializer="pickle").get()    
    return JsonResponse({"image": image_data})
        
@auth_required
def ShapInteraction(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    param = request.GET.get("param", None)
    metagenes = param.replace("\t", "").replace(" ", "").split(',')
    if len(metagenes)==0:
        return HttpResponse("No metagene", status=404)
    if len(metagenes)==1:
        if metagenes[0] not in usr.shapCache.feature_names:
            return HttpResponse("Illegal metagene", status=500)
        with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
            figure1 = io.BytesIO()
            shap.plots.scatter(usr.shapCache[:, metagenes[0]], color=usr.shapCache)
            plt.savefig(figure1, format="svg", bbox_inches="tight")
            plt.close()
    else:
        if metagenes[0] not in usr.shapCache.feature_names or metagenes[1] not in usr.shapCache.feature_names:
            return HttpResponse("Illegal metagene", status=500)
        with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
            figure1 = io.BytesIO()
            shap.plots.scatter(usr.shapCache[:, metagenes[0]], color=usr.shapCache[:, metagenes[1]])
            plt.savefig(figure1, format="svg", bbox_inches="tight")
            plt.close()
    image_data = base64.b64encode(figure1.getvalue()).decode("utf-8")
    return JsonResponse({"image": image_data})
        
@auth_required
def CFreport1(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)

    usr = checkRes["usrData"]

    try:
        params = json.loads(request.body)
        paramT = MLparamSetting(params.get('T_model', 'rf'), params.get('T_p', ''))
        paramY = MLparamSetting(params.get('Y_model', 'rf'), params.get('Y_p', ''))
        testRatio = float(params['testRatio'])
        X, y, T = shared_ml_dml_cf(usr, int(params['use_ICAcohort']), params['Y_log'], params['Y_type'], params['Y_label'], params['Y_interest'], float(params['drop']), params['T_label'], params['T_interest'])
        #print(X)
        #print(y)
        #print(T)
    except Exception as e:
        return HttpResponse(f"Illegal parameters: {e}", status=500)
 
    mlType = params.get('Y_type', 'classification')

    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=testRatio, stratify=T, random_state=42)
    try:
        model_func_map = {'rf': run_Rf, 'xgb': run_XGBoost, 'lightgbm': run_LightGBM, 'gb':run_GB}
        run_treatment = model_func_map.get(params['T_model'], run_Rf)
        resultT = run_treatment.apply_async(('classification', paramT, X, X_train, T_train, X_test, T_test, testRatio), serializer="pickle").get()

        run_outcome = model_func_map.get(params['Y_model'], run_Rf)
        resultY = run_outcome.apply_async((mlType, paramY, X, X_train, y_train, X_test, y_test, testRatio), serializer="pickle").get()

        model_t, model_y = resultT[1][0], resultY[1][0]
        performance_image = run_CFreport1.apply_async((mlType, model_t, model_y, X_test, T_test, y_test), serializer="pickle").get()
    except Exception as e:
        return HttpResponse(f"Failed running CF: {e}", status=500)
        
    usr.model_ty = (model_t, model_y)
    if usr.save() is False:
        return HttpResponse("Can't save user record", status=500)
    return render(request, "imf_template.html", {"svg_content": performance_image, "title":"Model Performance", "message":""})

@auth_required
def CFreport2(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    usr = checkRes["usrData"]
    if len(usr.model_ty)==0:
        return HttpResponse('Run T&Y model First.', status=400)
    try:
        params = json.loads(request.body)
        testRatio = float(params['testRatio'])
        mlType = params.get('Y_type', 'classification')
        cf_params = params.get('cf_params', '')
        cf_params = MLparamSetting('rf', cf_params)
        model_t, model_y = usr.model_ty
        if mlType == 'classification':
            causal_forest = CausalForestDML(model_t=model_t, model_y=model_y, discrete_treatment=True, discrete_outcome=True, **cf_params)
        else:
            causal_forest = CausalForestDML(model_t=model_t, model_y=model_y, discrete_treatment=True, discrete_outcome=False, **cf_params)
            
        X, y, T = shared_ml_dml_cf(usr, int(params['use_ICAcohort']), params['Y_log'], params['Y_type'], params['Y_label'], params['Y_interest'], float(params['drop']), params['T_label'], params['T_interest'])
        # print(X)
        # print(y)
        # print(T)
        X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=testRatio, stratify=T, random_state=42)
    except Exception as e:
        return HttpResponse(f"Error generating CFreport: {e}", status=500)

    causal_forest.fit(y_train, T_train, X=X_train)
    image = run_CFreport2.apply_async((causal_forest, X_test), serializer="pickle").get()
    usr.cf = (causal_forest, X, X_test.index.to_list())
    message = "Training ATE: "+str(causal_forest.ate_)
    message += "     ATE_STD_ERR: "+str(causal_forest.ate_stderr_)
    if usr.save() is False:
        return HttpResponse("Can't save user record", status=500)
    return render(request, "imf_template.html", {"svg_content": image, "title":"Treatment Result (Test dataset)", "message": message})
        
@auth_required  
def CFfeaImp(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    usr = checkRes["usrData"]
    if len(usr.cf)==0:
        return HttpResponse(f"Run Causal Forests first.", status=500)
    causal_forest = usr.cf[0]
    X, X_test = usr.cf[1], usr.cf[1].loc[usr.cf[2]]
    if SHAP_PLOT_DATASET=='X':
        shap_values = causal_forest.shap_values(X)
    elif SHAP_PLOT_DATASET=='X_train':
        shap_values = causal_forest.shap_values(X.drop(usr.cf[2]))
    else:
        shap_values = causal_forest.shap_values(X_test) 
    shap_val = shap_values['Y0']['T0_1']

    # Calculate spread (max - min) for each feature
    spread = np.ptp(shap_val.values, axis=0)  # np.ptp computes peak-to-peak (max - min)
    
    sorted_indices = np.argsort(spread)[::-1]
    figure = io.BytesIO()
    shap.plots.beeswarm(
        shap_val,
        max_display=35,
        show=False,
        order=sorted_indices
    )
    plt.savefig(figure, format="svg", bbox_inches="tight")
    plt.close()
    image = base64.b64encode(figure.getvalue()).decode("utf-8")
    usr.shapCache = shap_val
    if usr.save() is False:
        return HttpResponse("Can't save user record", status=500)
    return render(request, "shap_importance.html", {"svg_content": image,"clientID":usr.cID})

    