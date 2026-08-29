
EPITHELIAL_MALIGNANT_REST = {
                b'B_cell' : 'non_epithelial_malignant',
                b'Endothelial' : 'non_epithelial_malignant',
                b'Epithelial' : 'epithelial_malignant',
                b'Fibroblast' : 'non_epithelial_malignant',
                b'Malignant' : 'epithelial_malignant',
                b'Myeloid' : 'non_epithelial_malignant',
                b'Other_Stromal' : 'non_epithelial_malignant',
                b'Plasma'  : 'non_epithelial_malignant',
                b'T_NK_cell' : 'non_epithelial_malignant',
                b'Unknown' : 'non_epithelial_malignant'  }

SIMPLIFIED_BROAD = {
                b'B_cell' : 'Lymphoid',
                b'Endothelial' : 'Stromal',
                b'Epithelial' : 'Epithelial',
                b'Fibroblast' : 'Stromal',
                b'Malignant' : 'Malignant',
                b'Myeloid' : 'Myeloid',
                b'Other_Stromal' : 'Stromal',
                b'Plasma'  : 'Lymphoid',
                b'T_NK_cell' : 'Lymphoid',
                b'Neural/Glial': 'Stromal',
                b'Unknown' : 'Unknown'  }

IDENTITY = {
                b'B_cell': 'B_cell',
                b'Endothelial': 'Endothelial',
                b'Epithelial': 'Epithelial',
                b'Fibroblast': 'Fibroblast',
                b'Malignant': 'Malignant',
                b'Myeloid': 'Myeloid',
                b'Other_Stromal': 'Other_Stromal',
                b'Plasma': 'Plasma',
                b'T_NK_cell': 'T_NK_cell',
                b'Unknown': 'Unknown'
}

MALIGNANT_VS_REST = {
    
                b'B_cell' : 'non_malignant',
                b'Endothelial' : 'non_malignant',
                b'Epithelial' : 'non_malignant',
                b'Fibroblast' : 'non_malignant',
                b'Malignant' : 'epithelial_malignant',
                b'Myeloid' : 'non_malignant',
                b'Other_Stromal' : 'non_malignant',
                b'Plasma'  : 'non_malignant',
                b'T_NK_cell' : 'non_malignant',
                b'Unknown' : 'non_malignant'  
}


names_mapping = {
    'simplified_broad' : SIMPLIFIED_BROAD,
    'epithelial_malignant_vs_rest' : EPITHELIAL_MALIGNANT_REST,
    'identity': IDENTITY,
    'malignant_vs_rest' : MALIGNANT_VS_REST
}

def mapping_factory(name: str):
    return names_mapping[name]
