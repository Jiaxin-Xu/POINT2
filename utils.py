import random
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from collections import defaultdict

def set_global_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONASHSEED']=str(seed)
    print(f"Global random seed set to {seed}")


def visualize_important_nodes(smiles_list, node_importance_list, top_percent=20, output_dir='./'):
    """
    Visualize the top `top_percent` important nodes (atoms) in the molecules represented by SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings.
        node_importance_list (list): List of node importance values for each molecule.
        top_percent (float): The top percentage of nodes to highlight (e.g., 20 for top 20%).
        output_dir (str): Directory where the images will be saved.
    """
    for i, (smiles, node_importance) in enumerate(zip(smiles_list[:10], node_importance_list[:10])):  # Visualize only first 2
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            continue

        # Get indices of top 20% important atoms
        node_importance = np.array(node_importance)
        threshold = np.percentile(node_importance, 100 - top_percent)
        important_atoms = [j for j, imp in enumerate(node_importance) if imp >= threshold]
        
        # # Normalize importance values for coloring (scaling between 0 and 1)
        # min_importance = node_importance[important_atoms].min()
        # max_importance = node_importance[important_atoms].max()
        # normalized_importance = (node_importance[important_atoms] - min_importance) / (max_importance - min_importance)

        # # Assign shades of green based on normalized importance
        # highlight_colors = {
        #     idx: (0, intensity, 0, intensity)  # (R, G, B, Alpha)
        #     for idx, intensity in zip(important_atoms, normalized_importance)
        # }

        # # Highlight the important atoms
        # atom_highlight = {atom_idx: node_importance[atom_idx] for atom_idx in important_atoms}

        # Set up the drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
        # opts = drawer.drawOptions()
        
        # Highlight atom colors
        highlight_colors = {idx: (0.1, 1, 0.01) for idx in important_atoms}  # Green color for important atoms
        # highlight_colors = {idx: (1, 0, 0) for idx in important_atoms}  # Red color for important atoms

        # Draw the molecule
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=important_atoms, highlightAtomColors=highlight_colors)
        drawer.FinishDrawing()
        
        # Save the image
        img_path = os.path.join(output_dir, f"rationale_{i+1}.png")
        with open(img_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        print(f"Saved visualization for molecule {i+1} at {img_path}")

# Example usage
# visualize_important_nodes(smiles_list=smiles_test[:2], node_importance_list=node_importance_test, output_dir='./')


def extract_maccs_feature_descriptions_list():
    """
    Extracts descriptions for MACCS keys based on the comments provided in the dictionary.
    
    Returns:
        dict: A list with MACCS descriptions as values.
    """
    # print("Extract MACCS Fingerprint Feature Names ...")
    maccs_smarts = {
        0:('?', 0, 'DUMMY'),
        1:('?', 0, 'ISOTOPE'),
        #2:('[#104,#105,#106,#107,#106,#109,#110,#111,#112]',0),  # atomic num >103 Not complete
        2:('[#104]', 0, '#104limit'),  # limit the above def'n since the RDKit only accepts up to #104
        3:('[#32,#33,#34,#50,#51,#52,#82,#83,#84]', 0, 'Group_IVa_Va_VIa_Rows_4_6'), # Group IVa,Va,VIa Rows 4-6 
        4:('[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]',0, 'actinide'), # actinide
        5:('[Sc,Ti,Y,Zr,Hf]', 0,'Group_IIIB_IVB'), # Group IIIB,IVB (Sc...)  
        6:('[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]', 0, 'Lanthanide'), # Lanthanide
        7:('[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]', 0, 'Group_VB_VIB_VIIB'), # Group VB,VIB,VIIB
        8:('[!#6;!#1]1~*~*~*~1', 0, 'QAAA@1'), # QAAA@1
        9:('[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]', 0, 'Group_VIII'), # Group VIII (Fe...)
        10:('[Be,Mg,Ca,Sr,Ba,Ra]', 0, 'Group_IIa'), # Group IIa (Alkaline earth)
        11:('*1~*~*~*~1', 0, '4M_Ring'), # 4M Ring
        12:('[Cu,Zn,Ag,Cd,Au,Hg]', 0,'Group_IB_IIB'), # Group IB,IIB (Cu..)
        13:('[#8]~[#7](~[#6])~[#6]',0, 'ON(C)C'), # ON(C)C
        14:('[#16]-[#16]',0, 'S-S'), # S-S
        15:('[#8]~[#6](~[#8])~[#8]',0, 'OC(O)O'), # OC(O)O
        16:('[!#6;!#1]1~*~*~1',0, 'QAA@1'), # QAA@1
        17:('[#6]#[#6]',0, 'CTC'), #CTC
        18:('[#5,#13,#31,#49,#81]',0, 'Group_IIIA'), # Group IIIA (B...) 
        19:('*1~*~*~*~*~*~*~1',0, '7M_Ring'), # 7M Ring
        20:('[#14]',0, 'Si'), #Si
        21:('[#6]=[#6](~[!#6;!#1])~[!#6;!#1]',0, 'C=C(Q)Q'), # C=C(Q)Q
        22:('*1~*~*~1',0, '3M_Ring'), # 3M Ring
        23:('[#7]~[#6](~[#8])~[#8]',0, 'NC(O)O'), # NC(O)O
        24:('[#7]-[#8]',0, 'N-O'), # N-O
        25:('[#7]~[#6](~[#7])~[#7]',0, 'NC(N)N'), # NC(N)N
        26:('[#6]=;@[#6](@*)@*',0, 'C$=C($A)$A'), # C$=C($A)$A
        27:('[I]',0, 'I'), # I
        28:('[!#6;!#1]~[CH2]~[!#6;!#1]',0, 'QCH2Q'), # QCH2Q
        29:('[#15]',0, 'P'),# P
        30:('[#6]~[!#6;!#1](~[#6])(~[#6])~*',0, 'CQ(C)(C)A'), # CQ(C)(C)A
        31:('[!#6;!#1]~[F,Cl,Br,I]',0, 'QX'), # QX
        32:('[#6]~[#16]~[#7]',0, 'CSN'), # CSN
        33:('[#7]~[#16]',0, 'NS'), # NS
        34:('[CH2]=*',0, 'CH2=A'), # CH2=A
        35:('[Li,Na,K,Rb,Cs,Fr]',0, 'Group_IA'), # Group IA (Alkali Metal)
        36:('[#16R]',0, 'S_Heterocycle'), # S Heterocycle
        37:('[#7]~[#6](~[#8])~[#7]',0, 'NC(O)N'), # NC(O)N
        38:('[#7]~[#6](~[#6])~[#7]',0, 'NC(C)N'), # NC(C)N
        39:('[#8]~[#16](~[#8])~[#8]',0, 'OS(O)O'), # OS(O)O
        40:('[#16]-[#8]',0, 'S-O'), # S-O
        41:('[#6]#[#7]',0, 'CTN'), # CTN
        42:('F',0, 'F'), # F
        43:('[!#6;!#1;!H0]~*~[!#6;!#1;!H0]',0, 'QHAQH'), # QHAQH
        44:('?',0, 'OTHER'), # OTHER
        45:('[#6]=[#6]~[#7]',0, 'C=CN'), # C=CN
        46:('Br',0, 'Br'), # BR
        47:('[#16]~*~[#7]',0, 'SAN'), # SAN
        48:('[#8]~[!#6;!#1](~[#8])(~[#8])',0, 'OQ(O)O'), # OQ(O)O
        49:('[!+0]',0, 'CHARGE'), # CHARGE  
        50:('[#6]=[#6](~[#6])~[#6]',0, 'C=C(C)C'), # C=C(C)C
        51:('[#6]~[#16]~[#8]',0, 'CSO'), # CSO
        52:('[#7]~[#7]',0, 'NN'), # NN
        53:('[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]',0, 'QHAAAQH'), # QHAAAQH
        54:('[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]',0, 'QHAAQH'), # QHAAQH
        55:('[#8]~[#16]~[#8]',0, 'OSO'), #OSO
        56:('[#8]~[#7](~[#8])~[#6]',0, 'ON(O)C'), # ON(O)C
        57:('[#8R]',0, 'O_Heterocycle'), # O Heterocycle
        58:('[!#6;!#1]~[#16]~[!#6;!#1]',0, 'QSQ'), # QSQ
        59:('[#16]!:*:*',0, 'Snot%A%A'), # Snot%A%A
        60:('[#16]=[#8]',0, 'S=O'), # S=O
        61:('*~[#16](~*)~*',0, 'AS(A)A'), # AS(A)A
        62:('*@*!@*@*',0, 'A$!A$A'), # A$!A$A
        63:('[#7]=[#8]',0, 'N=O'), # N=O
        64:('*@*!@[#16]',0, 'A$A!S'), # A$A!S
        65:('c:n',0, 'C%N'), # C%N
        66:('[#6]~[#6](~[#6])(~[#6])~*',0, 'CC(C)(C)A'), # CC(C)(C)A
        67:('[!#6;!#1]~[#16]',0, 'QS'), # QS
        68:('[!#6;!#1;!H0]~[!#6;!#1;!H0]',0, 'QHQH'), # QHQH (&...) SPEC Incomplete
        69:('[!#6;!#1]~[!#6;!#1;!H0]',0, 'QQH'), # QQH
        70:('[!#6;!#1]~[#7]~[!#6;!#1]',0, 'QNQ'), # QNQ
        71:('[#7]~[#8]',0, 'NO'), # NO
        72:('[#8]~*~*~[#8]',0, 'OAAO'), # OAAO
        73:('[#16]=*',0, 'S=A'), # S=A
        74:('[CH3]~*~[CH3]',0, 'CH3ACH3'), # CH3ACH3
        75:('*!@[#7]@*',0, 'A!N$A'), # A!N$A
        76:('[#6]=[#6](~*)~*',0, 'C=C(A)A'), # C=C(A)A
        77:('[#7]~*~[#7]',0, 'NAN'), # NAN
        78:('[#6]=[#7]',0, 'C=N'), # C=N
        79:('[#7]~*~*~[#7]',0, 'NAAN'), # NAAN
        80:('[#7]~*~*~*~[#7]',0, 'NAAAN'), # NAAAN
        81:('[#16]~*(~*)~*',0, 'SA(A)A'), # SA(A)A
        82:('*~[CH2]~[!#6;!#1;!H0]',0, 'ACH2QH'), # ACH2QH
        83:('[!#6;!#1]1~*~*~*~*~1',0, 'QAAAA@1'), # QAAAA@1
        84:('[NH2]',0, 'NH2'), #NH2
        85:('[#6]~[#7](~[#6])~[#6]',0, 'CN(C)C'), # CN(C)C
        86:('[C;H2,H3][!#6;!#1][C;H2,H3]',0, 'CH2QCH2'), # CH2QCH2
        87:('[F,Cl,Br,I]!@*@*',0, 'X!A$A'), # X!A$A
        88:('[#16]',0, 'S'), # S
        89:('[#8]~*~*~*~[#8]',0, 'OAAAO'), # OAAAO
        90:('[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]',0, 'QHAACH2A'), # QHAACH2A
        91:('[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]',0, 'QHAAACH2A'), # QHAAACH2A
        92:('[#8]~[#6](~[#7])~[#6]',0, 'OC(N)C'), # OC(N)C
        93:('[!#6;!#1]~[CH3]',0, 'QCH3'), # QCH3
        94:('[!#6;!#1]~[#7]',0, 'QN'), # QN
        95:('[#7]~*~*~[#8]',0, 'NAAO'), # NAAO
        96:('*1~*~*~*~*~1',0, '5M_Ring'), # 5 M ring
        97:('[#7]~*~*~*~[#8]',0, 'NAAAO'), # NAAAO
        98:('[!#6;!#1]1~*~*~*~*~*~1',0, 'QAAAAA@1'), # QAAAAA@1
        99:('[#6]=[#6]',0, 'C=C'), # C=C
        100:('*~[CH2]~[#7]',0, 'ACH2N'), # ACH2N
        101:('[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]',0, '8M_14M_Ring'), # 8M Ring or larger. This only handles up to ring sizes of 14
        102:('[!#6;!#1]~[#8]',0, 'QO'), # QO
        103:('Cl',0, 'Cl'), # CL
        104:('[!#6;!#1;!H0]~*~[CH2]~*',0, 'QHACH2A'), # QHACH2A
        105:('*@*(@*)@*',0, 'A$A($A)$A'), # A$A($A)$A
        106:('[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]',0, 'QA(Q)Q'), # QA(Q)Q
        107:('[F,Cl,Br,I]~*(~*)~*',0, 'XA(A)A'), # XA(A)A
        108:('[CH3]~*~*~*~[CH2]~*',0, 'CH3AAACH2A'), # CH3AAACH2A
        109:('*~[CH2]~[#8]',0, 'ACH2O'), # ACH2O
        110:('[#7]~[#6]~[#8]',0, 'NCO'), # NCO
        111:('[#7]~*~[CH2]~*',0, 'NACH2A'), # NACH2A
        112:('*~*(~*)(~*)~*',0, 'AA(A)(A)A'), # AA(A)(A)A
        113:('[#8]!:*:*',0, 'Onot%A%A'), # Onot%A%A
        114:('[CH3]~[CH2]~*',0, 'CH3CH2A'), # CH3CH2A
        115:('[CH3]~*~[CH2]~*',0, 'CH3ACH2A'), # CH3ACH2A
        116:('[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]',0, 'CH3AACH2A'), # CH3AACH2A
        117:('[#7]~*~[#8]',0, 'NAO'), # NAO
        118:('[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]',1, 'ACH2CH2A>1'), # ACH2CH2A > 1
        119:('[#7]=*',0, 'N=A'), # N=A
        120:('[!#6;R]',1, 'Heterocyclic_atom>1'), # Heterocyclic atom > 1 (&...) Spec Incomplete
        121:('[#7;R]',0, 'N_Heterocycle'), # N Heterocycle
        122:('*~[#7](~*)~*',0, 'AN(A)A'), # AN(A)A
        123:('[#8]~[#6]~[#8]',0, 'OCO'), # OCO
        124:('[!#6;!#1]~[!#6;!#1]',0, 'QQ'), # QQ
        125:('?',0, 'Aromatic_Ring>1'), # Aromatic Ring > 1
        126:('*!@[#8]!@*',0, 'A!O!A'), # A!O!A
        127:('*@*!@[#8]',1, 'A$A!O>1'), # A$A!O > 1 (&...) Spec Incomplete
        128:('[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]',0, 'ACH2AAACH2A'), # ACH2AAACH2A
        129:('[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]',0, 'ACH2AACH2A'), # ACH2AACH2A
        130:('[!#6;!#1]~[!#6;!#1]',1, 'QQ>1'), # QQ > 1 (&...)  Spec Incomplete
        131:('[!#6;!#1;!H0]',1, 'QH>1'), # QH > 1
        132:('[#8]~*~[CH2]~*',0, 'OACH2A'), # OACH2A
        133:('*@*!@[#7]',0, 'A$A!N'), # A$A!N
        134:('[F,Cl,Br,I]',0, 'HALOGEN'), # X (HALOGEN)
        135:('[#7]!:*:*',0, 'Nnot%A%A'), # Nnot%A%A
        136:('[#8]=*',1, 'O=A>1'), # O=A>1 
        137:('[!C;!c;R]',0, 'Heterocycle'), # Heterocycle
        138:('[!#6;!#1]~[CH2]~*',1, 'QCH2A>1'), # QCH2A>1 (&...) Spec Incomplete
        139:('[O;!H0]',0, 'OH'), # OH
        140:('[#8]',3, 'O>3'), # O > 3 (&...) Spec Incomplete
        141:('[CH3]',2, 'CH3>2'), # CH3 > 2  (&...) Spec Incomplete
        142:('[#7]',1, 'N>1'), # N > 1
        143:('*@*!@[#8]',0, 'A$A!O'), # A$A!O
        144:('*!:*:*!:*',0, 'Anot%A%Anot%A'), # Anot%A%Anot%A
        145:('*1~*~*~*~*~*~1',1, '6M_Ring>1'), # 6M ring > 1
        146:('[#8]',2, 'O>2'), # O > 2
        147:('[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]',0, 'ACH2CH2A'), # ACH2CH2A
        148:('*~[!#6;!#1](~*)~*',0, 'AQ(A)A'), # AQ(A)A
        149:('[C;H3,H4]',1, 'CH3>1'), # CH3 > 1
        150:('*!@*@*!@*',0, 'A!A$A!A'), # A!A$A!A
        151:('[#7;!H0]',0, 'NH'), # NH
        152:('[#8]~[#6](~[#6])~[#6]',0, 'OC(C)C'), # OC(C)C
        153:('[!#6;!#1]~[CH2]~*',0, 'QCH2A'), # QCH2A
        154:('[#6]=[#8]',0, 'C=O'), # C=O
        155:('*!@[CH2]!@*',0, 'A!CH2!A'), # A!CH2!A
        156:('[#7]~*(~*)~*',0, 'NA(A)A'), # NA(A)A
        157:('[#6]-[#8]',0, 'C-O'), # C-O
        158:('[#6]-[#7]',0, 'C-N'), # C-N
        159:('[#8]',1, 'O>1'), # O>1
        160:('[C;H3,H4]',0, 'CH3'), #CH3
        161:('[#7]',0, 'N'), # N
        162:('a',0, 'Aromatic'), # Aromatic
        163:('*1~*~*~*~*~*~1',0, '6M_Ring'), # 6M Ring
        164:('[#8]',0, 'O'), # O
        165:('[R]',0, 'Ring'), # Ring
        166:('?',0, 'Fragments'), # Fragments  FIX: this can't be done in SMARTS
        }
    feature_descriptions_ls = []
    for key, value in maccs_smarts.items():
        # Extract the comment from the tuple, which is the last part of the tuple
        description = value[2] if isinstance(value, tuple) and len(value) > 2 else ""
        # The description is generally after the '#' in the comment

        feature_descriptions_ls.append(description)
    
    return feature_descriptions_ls


def visualize_morgan_bits(smiles_list, radius=2, n_bits=2048, molsPerRow=4):
    bit_structures = defaultdict(set)  # Stores unique subgraphs for each bit across all molecules

    # Generate fingerprints and collect subgraphs for each bit
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        fp, bi = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=dict())
        for bit in fp.GetOnBits():
            # Extract subgraph for each bit
            submol = Chem.PathToSubmol(mol, [x[0] for x in bi[bit]])
            smi = Chem.MolToSmiles(submol)
            bit_structures[bit].add(smi)  # Add unique structure

    # Prepare to visualize
    list_bits = []
    legends = []

    # Collect all unique structures per bit
    for bit, subsmiles_set in bit_structures.items():
        for subsmi in subsmiles_set:
            submol = Chem.MolFromSmiles(subsmi)
            list_bits.append(submol)
            legends.append(f"Bit {bit}: {subsmi}")

    # Draw all bits with their corresponding structures
    img = Draw.MolsToGridImage(list_bits, molsPerRow=molsPerRow, legends=legends, subImgSize=(200, 200))
    return img

def remove_and_sort_duplicates(legends, list_bits):
    unique_pairs = {}
    
    for idx, legend in enumerate(legends):
        legend_int = int(legend)  # Convert string to integer for correct numeric comparison
        if legend_int not in unique_pairs:
            unique_pairs[legend_int] = list_bits[idx]
    
    # Sort by legend integer keys and keep corresponding list_bits
    sorted_legends = sorted(unique_pairs.keys())
    sorted_list_bits = [unique_pairs[legend] for legend in sorted_legends]
    
    return [str(legend) for legend in sorted_legends], sorted_list_bits