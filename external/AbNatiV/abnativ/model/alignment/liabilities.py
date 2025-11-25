"""
 Copyright 2023. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
"""


import numpy

chemical_liabilities = {
# HD : sequence motif,  specific location only, location type  liability score Int , fine liability score, reference
'NxS' : [ 'N-glycosylation', None,'any', 1 , 0.5 , 'Jarasch, A. et al. Developability Assessment During the Selection of Novel Therapeutic Antibodies. J. Pharm. Sci. 104, (2015).' ],
'NxT' : [ 'N-glycosylation', None,'any', 1 , 0.5 , 'Jarasch, A. et al. Developability Assessment During the Selection of Novel Therapeutic Antibodies. J. Pharm. Sci. 104, (2015).' ],
#Prone to deamidation, asparagine and aspartic acid in more flexible regions, such as the CDR loops, are more prone, respectively, to deamidation and isomerisation than those found in rigid regions, such as the framework
'SNG' : [ 'deamidation',     None,'any', 2 ,  1 , 'Chelius D, Rehder DS, Bondarenko PV Identification and characterization of deamidation sites in the conserved regions of human immunoglobulin antibodies. Anal Chem. 2005'],
'ENN' : [ 'deamidation',     None,'any', 1 ,  0.6 , 'Chelius D, Rehder DS, Bondarenko PV Identification and characterization of deamidation sites in the conserved regions of human immunoglobulin antibodies. Anal Chem. 2005'],
'LNG' : [ 'deamidation',     None,'any', 2 ,  1 , 'Chelius D, Rehder DS, Bondarenko PV Identification and characterization of deamidation sites in the conserved regions of human immunoglobulin antibodies. Anal Chem. 2005'],
'LNN' : [ 'deamidation',     None,'any', 1 ,  0.6 , 'Chelius D, Rehder DS, Bondarenko PV Identification and characterization of deamidation sites in the conserved regions of human immunoglobulin antibodies. Anal Chem. 2005'],
# Isomerization, DG is expected to be more labile than DS, and DS more so than DD or DT
'DG' : [ 'isomerization',     None,'any', 1 ,  1 , 'Lu et al., Deamidation and isomerization liability analysis of 131 clinical-stage antibodies 2019'],
'DS' : [ 'isomerization',     None,'any', 1 ,  0.7 , 'Lu et al., Deamidation and isomerization liability analysis of 131 clinical-stage antibodies 2019'],
'DT' : [ 'isomerization',     None,'any', 0.5 ,  0.5, 'Lu et al., Deamidation and isomerization liability analysis of 131 clinical-stage antibodies 2019'],
'DD' : [ 'isomerization',     None,'any', 0.5 ,  0.5 , 'Lu et al., Deamidation and isomerization liability analysis of 131 clinical-stage antibodies 2019'],
'DH' : [ 'isomerization',     None,'any', 0.5 ,  0.5 , 'Lu et al., Deamidation and isomerization liability analysis of 131 clinical-stage antibodies 2019'],
# Asparagine residues followed by certain sequence motifs including glycine, serine, threonine, aspartate and histidine have been reported as degradation hotspots
'NG' : [ 'deamidation',     None,'any', 1 ,  0.7 , 'Lu et al., Deamidation and isomerization liability analysis of 131 clinical-stage antibodies 2019'],
'NS' : [ 'deamidation',     None,'any', 0.5 ,  0.3 , 'Lu et al., Deamidation and isomerization liability analysis of 131 clinical-stage antibodies 2019'],
'NT' : [ 'deamidation',     None,'any', 0.5 ,  0.2 , 'Lu et al., Deamidation and isomerization liability analysis of 131 clinical-stage antibodies 2019'],
'NH' : [ 'deamidation',     None,'any', 0.5 ,  0.2 , 'Lu et al., Deamidation and isomerization liability analysis of 131 clinical-stage antibodies 2019'],
'ND' : [ 'deamidation',     None,'any', 0.5 ,  0.2 , 'Lu et al., Deamidation and isomerization liability analysis of 131 clinical-stage antibodies 2019']
}

cdr_liabilities = {
# HD : liability, sequence motif,  specific location only,  liability score Int , fine liability score incremental, reference
# Cross-interaction: RS highest, WS less, YG even less
'RS' : [ 'poor-specificity',     'CDR','CDR', 1 ,  0.9, 'Kelly et al., 2018 Reduction of Nonspecificity Motifs in Synthetic Antibody Libraries'],
'WS' : [ 'poor-specificity',     'CDR','CDR', 1 ,  0.8, 'Kelly et al., 2018 Reduction of Nonspecificity Motifs in Synthetic Antibody Libraries'],
'YG' : [ 'poor-specificity',     'CDR','CDR', 0.5 ,  0.5, 'Kelly et al., 2018 Reduction of Nonspecificity Motifs in Synthetic Antibody Libraries'],
# Enrichment in CDR H3 causes sometimes but rarely non-specificty. Glycine contributes to flexibility of H3 loop, which may allow greater olasticity and broader antigen recognition
'n3G' : [ 'poor-specificity',   'CDRH3','CDR', 0.5 ,  0.3, 'Kelly et al., 2018 Reduction of Nonspecificity Motifs in Synthetic Antibody Libraries'], # n2 means 2 or more
'n2V' : [ 'poor-specificity',   'CDRH3','CDR', 0.5 ,  0.2, 'Kelly et al., 2018 Reduction of Nonspecificity Motifs in Synthetic Antibody Libraries'],
'n1W' : [ 'poor-specificity',   'CDRH3','CDR', 1 ,  0.6, 'Kelly et al., 2018 Reduction of Nonspecificity Motifs in Synthetic Antibody Libraries'],
'n1C' : [ 'disulphide-dimerisation','CDR','CDR',1,  1  , 'Kelly et al., 2018 Reduction of Nonspecificity Motifs in Synthetic Antibody Libraries'],
'n1M' : [ 'oxydation','CDR','CDR',1,  1  , ''],
'n2R' : [ 'poor-specificity',   'CDRH3','CDR', 0.5 ,  0.3, 'Kelly et al., 2018 Reduction of Nonspecificity Motifs in Synthetic Antibody Libraries'],
# The best indicators of antibody specificity in terms of CDR amino acid composition are reduced levels of arginine and lysine. Lysines are very rare in CDRs, but could potentially be a glycation site and reduced activity
'n2K' : [ 'poor-specificity/glycation','CDR','CDR',0.5,1  , 'Kelly et al., 2018 Reduction of Nonspecificity Motifs in Synthetic Antibody Libraries']
#K49    Glycation hotspot    CDR2LC    https://pubmed.ncbi.nlm.nih.gov/18307322/
#K98    Glycation hotspot    CDR3HC    https://pubmed.ncbi.nlm.nih.gov/21287557/
#M4    Critical quality attributes.  Kabat numbering. Employed  specific stress conditions to antibody, elevated temperatures, pH, oxidizing agents, and forced glycation with glucose incubation. However, none of the assessed degradation products led to a complete loss of functionality if only one light or heavy chain of the native antibody was affected    LC    https://pubmed.ncbi.nlm.nih.gov/24441081/
#N30/31    Critical quality attributes.  Kabat numbering. Employed  specific stress conditions to antibody, elevated temperatures, pH, oxidizing agents, and forced glycation with glucose incubation. However, none of the assessed degradation products led to a complete loss of functionality if only one light or heavy chain of the native antibody was affected    LC    https://pubmed.ncbi.nlm.nih.gov/24441081/
#N92    Critical quality attributes.  Kabat numbering. Employed  specific stress conditions to antibody, elevated temperatures, pH, oxidizing agents, and forced glycation with glucose incubation. However, none of the assessed degradation products led to a complete loss of functionality if only one light or heavy chain of the native antibody was affected    LC    https://pubmed.ncbi.nlm.nih.gov/24441081/
#M100c    Critical quality attributes.  Kabat numbering. Employed  specific stress conditions to antibody, elevated temperatures, pH, oxidizing agents, and forced glycation with glucose incubation. However, none of the assessed degradation products led to a complete loss of functionality if only one light or heavy chain of the native antibody was affected    HC     https://pubmed.ncbi.nlm.nih.gov/24441081/
#K33    Critical quality attributes. Kabat numbering. Employed  specific stress conditions to antibody, elevated temperatures, pH, oxidizing agents, and forced glycation with glucose incubation. However, none of the assessed degradation products led to a complete loss of functionality if only one light or heavy chain of the native antibody was affected    HC    https://pubmed.ncbi.nlm.nih.gov/24441081/
#H54    Residues account for most chemical modifications in 121 clinical antibody sequences, Kabat-Chotia numbering
#H98    Residues account for most chemical modifications in 121 clinical antibody sequences, Kabat-Chotia numbering
#L30    Residues account for most chemical modifications in 121 clinical antibody sequences, Kabat-Chotia numbering
}
