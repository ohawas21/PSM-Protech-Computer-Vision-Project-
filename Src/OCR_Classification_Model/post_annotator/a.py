rea=r'C:\Users\Admin\PSE\PSM-Protech-Feasibility-Study\Src\OCR_Classification_Model\post_annotator\form_surface_profile copy 2.png'
a=rea.find('surface')
print(a)
with open(rea,'r') as file:
    file.write('surface.png')