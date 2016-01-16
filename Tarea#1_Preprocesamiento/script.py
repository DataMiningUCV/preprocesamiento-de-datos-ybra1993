# -*- coding: utf-8 -*-
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib as matplot
import re
input_data_path = 'data/data_input.csv' #Data de entrada
output_data_path = 'data/output_data.csv' #Data de salida - Vista Minable
header_name = ['periodRenew','id','birthDate','age','civilStatus','gender','school',
              'admissionYear','admissionForm','coursesSemester','isChangeAddress',
              'reasonChange','coursesEnrolled','coursesApproved','coursesRemoved','coursesFailed',
              'weightedAverage','efficiency','coursesFailedReason','coursesCurrent',
              'isThesisEnroll','thesisEnrolled','origin','residency','roomies',
              'houseType','isRent','addressHouse','isMarried','isBenefitUniversity',
              'reasonAndYear','isEconomicActivity','typeEcoActivity','scholarship',
              'contributionHouseholder','contributionFamily','contributionActivities',
              'contributionMonthlyTotal','foodExpenses','transportExpenses','medicalExpenses','dentalExpenses',
              'personalExpenses','rentExpenses','studyMaterialExpenses','recreationalExpenses',
              'otherExpenses','totalExpenses','howHouseholder','familyBurden','incomeHouseholder',
              'incomeOther','incomeTotal','houseExpensesHouseholder','foodExpensesHouseholder',
              'transportExpensesHouseholder','medicalExpensesHouseholder','dentalExpensesHouseholder',
              'studyExpensesHouseholder','servicesExpensesHouseholder','condominiumExpensesHouseholder',
              'otherExpensesHouseholder','totalExpensesHouseholder','raiting','reviewsUsers']

#Cargando los datos
dataframe = pd.read_csv(input_data_path,skiprows=1,header=None,encoding='utf-8')
output_dataframe = pd.DataFrame(columns = header_name)

#Limpiando el periodo a renovar
period_messy = dataframe[1].str.replace('[ \\\\-]','')
period = period_messy.str.extract('^(?P<year>(?:20)?\d{2})(?P<nro>pri(?:mero)?|seg(?:undo)?|ii?|0?[12]s?)$', re.IGNORECASE)
period_p2 = period_messy.str.extract('^(?P<nro>pri(?:mero)?|seg(?:undo)?|ii?|0?[12]s?)(?P<year>(?:20)?\d{2})$', re.IGNORECASE)

period.update(period_p2)

period.nro = period.nro.str.replace('pri(mero)?|i|0?1s?', '1', flags=re.IGNORECASE).str.replace('seg(undo)?|ii|0?2s?', '2', flags=re.IGNORECASE)
period.year = period.year.str.replace('^\d{2}$', lambda str: '20'+str)

#Limpiando el id por si acaso no hay repetidos
output_dataframe.id = dataframe[2].drop_duplicates()

#Limpiando la Fecha de Nacimiento
dataframe[3] = dataframe[3].str.replace('[\s|/]','-').str.replace('^\d+\-\d+\-\d{2}$',lambda str: str.group(0)[:-2]+'19'+str.group(0)[-2:])
output_dataframe.birthDate = pd.to_datetime(dataframe[3],errors='coerce',dayfirst=True) #formato yyyy-mm-dd
output_dataframe.birthDate = output_dataframe.birthDate.fillna(output_dataframe.birthDate.mode().iloc[1])
#Limpiando la Edad
output_dataframe.age = dataframe[4].str.extract('(\d{2})',re.IGNORECASE).astype('int')

#Limpiando el Edo Civil
output_dataframe.civilStatus = dataframe[5].replace(['Viudo (a)','Casado (a)','Soltero (a)','Unido (a)'],[0,1,2,np.nan])
output_dataframe.civilStatus = output_dataframe.civilStatus.fillna(output_dataframe.civilStatus.mode().iloc[0]).astype('int')

#Limpiando el Sexo y la Escuela
output_dataframe.gender = dataframe[6].replace(['Femenino', 'Masculino'], [0,1])
output_dataframe.school = dataframe[7].replace([u'Enfermería',u'Bioanálisis'],[0,1])

#El ano de ingreso lo dejamos igual
output_dataframe.admissionYear = dataframe[8]

#Limpiando el modo de ingreso
output_dataframe.admissionForm = dataframe[9].replace([r'Asignado.*', r'Convenios.*',r'Prueba.*'], [0,1,2], regex=True).astype('int')

#Limpiando el Semestre que cursa
output_dataframe.coursesSemester = dataframe[10].str.extract('^(\d{1,2})',re.IGNORECASE)

#Limpiando las columnas con valores Binarios (SI|NO)
output_dataframe.isChangeAddress = dataframe[11].replace(['Si','No'],[1,0]).astype('int')
output_dataframe.isThesisEnroll = dataframe[21].replace(['Si','No'],[1,0]).astype('int')
output_dataframe.isMarried = dataframe[29].replace(['Si','No'],[1,0]) .astype('int')
output_dataframe.isBenefitUniversity = dataframe[30].replace(['Si','No'],[1,0]).astype('int')
output_dataframe.isEconomicActivity = dataframe[32].replace(['Si','No'],[1,0]).astype('int')

#Limpiando el motivo de mudanza
output_dataframe.reasonChange = dataframe[12].fillna('NA')

#El numero de materias inscritas la dejamos igual
output_dataframe.coursesEnrolled = dataframe[13].astype('int')

#Limpiando el numero de materias aprobadas
output_dataframe.coursesApproved = dataframe[14].str.extract('(\.*\d+)',re.IGNORECASE).astype('int')

#El numero de materias retiradas la dejamos igual
output_dataframe.coursesRemoved = dataframe[15].astype('int')

#El numero de materias aplazadas la dejamos igual
output_dataframe.coursesFailed = dataframe[16]

#Limpiando la columna de promedio ponderado
output_dataframe.weightedAverage = dataframe[17].astype('string').str.replace('\.','').str.replace('(\d)+', lambda str : str.group(0)[:1]+'.'+str.group(0)[1:] if int(str.group(0)[:1])>2 else str.group(0)[:2]+'.'+str.group(0)[2:])

#Limpiando la columna de eficiencia
output_dataframe.efficiency = dataframe[18].astype('string').str.replace('\.','').str.replace('(\d)+', lambda str : '0.'+str.group(0)[1:] if int(str.group(0)[:1])>1 else str.group(0)[:1]+'.'+str.group(0)[1:])

#Limpiando la columna motivo de reprobacion
output_dataframe.coursesFailedReason = dataframe[19].fillna('NA')

#Limpiando la columna de cuantas veces ha inscrito la tesis

output_dataframe.thesisEnrolled = dataframe[22].fillna(0).replace([r'P.+',r'S.+',r'M.+'],[1,2,3],regex=True)

#Limpiando la columna de procedencia

output_dataframe.origin = dataframe[23].replace([r'.*Libertador+.*',r'.*(Sucre|Baruta|El Hatillo|Chacao|Altos|Guarenas|Valles|Barlovento).*',r'Ara.*',r'Apu.*',ur'Tác.*',r'Var.*',r'Mon.*',r'Por.*',r'Nue*.',r'Tru*.',r'Lar.*',r'Bol.*',r'Bar.*',r'Suc.*',r'Anz.*',ur'Mér.*',r'Delta.*',r'Yar.*',ur'Guár.*'],range(19),regex=True).astype('int')

#Limpiando la columna de residencia

output_dataframe.residency = dataframe[24].replace([r'.*Libertador+.*',r'.*Sucre',r'.*Baruta',r'.*El Hatillo',r'.*Chacao',r'.*Altos',r'.*Guarenas',r'.*Valles'],range(8),regex=True)
output_dataframe.residency = output_dataframe.residency.fillna(output_dataframe.residency.mode().iloc[0]).astype('int')


output_dataframe.to_csv(output_data_path, encoding='utf-8',index=False)