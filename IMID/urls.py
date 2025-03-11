from django.urls import path, re_path
from . import views

urlpatterns = [
    # path("", views.index, name="index"),
    path("accounts/rest/login/", views.restLogin, name="restLogin"),
    path("GeneLookup/", views.GeneLookup, name="GeneLookup"),
    path("geneExpression/", views.opExpression, name="op_expression"),
    path("meta/", views.opMeta, name="op_meta"),
    path("eda/integrate/", views.edaIntegrate, name="edaIntegrate"),
    path("eda/", views.eda, name="eda"),
    path("dgea/candiGenes/", views.candiGenes, name="candiGenes"),
    path("dgea/", views.dgea, name="dgea"),
    path("goenrich/", views.goenrich, name="goenrich"),
    path("", views.tab, name="tab"),
    path("meta/columns/", views.meta_columns, name="meta_columns"),
    path(
        r"meta/<slug:colName>/",
        views.meta_column_values,
        name="meta_columns_values",
    ),
    path(
        "processedFile/corrected/", views.downloadCorrected, name="downloadCorrected"
    ),
    path("user/", views.checkUser, name="checkUser"),
    path(
        "processedFile/correctedCluster/",
        views.downloadCorrectedCluster,
        name="downloadCorrectedCluster",
    ),
    path("ica/metagenes/", views.downloadICA, name="downloadICA"),
    path("compType/ICA/report/", views.ICAreport, name="ICAreport"),
    path("compType/ORA/report/", views.ORAreport, name="ORAreport"),
    path("compType/ICA/", views.compICA, name="compICA"),
    path("compType/ICA/ica_cohort/", views.compICAcohort, name="compICAcohort"),
    path("compType/ICA/ica_all/", views.compICAall, name="compICAall"),
    path("compType/ORA/", views.compORA, name="compORA"),
    path("compType/ORA/report/DE/", views.ORADE, name="ORAreport"),
    path("ml/featureImportance/data/", views.feaImpData, name="feaImpData"),
    path("ml/featureImportance/", views.feaImp, name="feaImp"),
    path("ml/featureDependence/", views.ShapInteraction, name="ShapInteraction"),
    path("ml/<slug:mlType>/<slug:mlMethod>/", views.goML, name="runML"),
    path("dml/featureImportance/<int:number>/", views.DMLfeaImp, name="DMLfeaImp"),
    path("dml/report/", views.DMLreport, name="DMLreport"),
    path("dml/report/distribution/",views.DMLreportDis, name="DMLreportDis"),
    path("dml/<slug:mlType>/<slug:mlMethod>/<int:number>/", views.goDML, name="runDML"),
    path("cf/report/TYmodels/", views.CFreport1, name="CFreport1"),
    path("cf/report/causalForests/", views.CFreport2, name="CFreport2"),
    path("cf/featureImportance/", views.CFfeaImp, name="CFfeaImp")
]
