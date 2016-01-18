# -*- coding: utf-8 -*-
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib as matplot
import re
input_data_path = 'data/data_input.csv' #Data de entrada
output_data_path = 'data/output_data.csv' #Data de salida - Vista Minable
header_name = ['periodYearRenew', 'periodNumberRenew','id','birthDate','civilStatus','gender','school',
              'admissionYear','admissionForm','coursesSemester',
              'coursesEnrolled','coursesApproved','coursesRemoved','coursesFailed',
              'weightedAverage','efficiency','coursesCurrent',
              'isThesisEnroll','thesisEnrolled','origin','residency','roomies',
              'houseType','isChangeAddress','isMarried','isBenefitUniversity','isEconomicActivity','scholarship',
              'contributionHouseholder','contributionFamily','contributionActivities',
              'contributionMonthlyTotal','foodExpenses','transportExpenses','medicalExpenses','dentalExpenses',
              'personalExpenses','rentExpenses','studyMaterialExpenses','recreationalExpenses',
              'otherExpenses','totalExpenses','howHouseholder','familyBurden','incomeHouseholder',
              'incomeOther','incomeTotal','houseExpensesHouseholder','foodExpensesHouseholder',
              'transportExpensesHouseholder','medicalExpensesHouseholder','dentalExpensesHouseholder',
              'studyExpensesHouseholder','servicesExpensesHouseholder','condominiumExpensesHouseholder',
              'otherExpensesHouseholder','totalExpensesHouseholder','raiting']

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

output_dataframe.periodNumberRenew = period.nro.fillna(period.nro.mode().iloc[0])
output_dataframe.periodYearRenew = period.year.fillna(period.year.mode().iloc[0]).astype('int')

#Limpiando el id por si acaso no hay repetidos
output_dataframe.id = dataframe[2].drop_duplicates()

#Limpiando la Fecha de Nacimiento
dataframe[3] = dataframe[3].str.replace('[\s|/]','-').str.replace('^\d+\-\d+\-\d{2}$',lambda str: str.group(0)[:-2]+'19'+str.group(0)[-2:])
output_dataframe.birthDate = pd.to_datetime(dataframe[3],errors='coerce',dayfirst=True) #formato yyyy-mm-dd
output_dataframe.birthDate = output_dataframe.birthDate.fillna(output_dataframe.birthDate.mode().iloc[1])

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

#El numero de materias inscritas la dejamos igual
output_dataframe.coursesEnrolled = dataframe[13].astype('int')

#Limpiando el numero de materias aprobadas
output_dataframe.coursesApproved = dataframe[14].str.extract('(\.*\d+)',re.IGNORECASE).astype('int')

#El numero de materias retiradas la dejamos igual
output_dataframe.coursesRemoved = dataframe[15].astype('int')

#El numero de materias aplazadas la dejamos igual
output_dataframe.coursesFailed = dataframe[16]

#Limpiando la columna de promedio ponderado
output_dataframe.weightedAverage = dataframe[17].apply(lambda x: float(x) if x>=0 and x<=20 else (float(x)/1000.0 if x>=1000 and x<=20000 else np.nan))

#Limpiando la columna de eficiencia
output_dataframe.efficiency = dataframe[18].apply(lambda x: float(x) if x>=0 and x<=1 else (float(x)/10000.0 if x>=1000 and x<=10000 else np.nan))

#El numero de materias inscritas en el semestre actual lo dejamos igual
output_dataframe.coursesCurrent = dataframe[20].astype('int')

#Limpiando la columna de cuantas veces ha inscrito la tesis

output_dataframe.thesisEnrolled = dataframe[22].fillna(0).replace([r'P.+',r'S.+',r'M.+'],[1,2,3],regex=True)

#Limpiando la columna de procedencia

output_dataframe.origin = dataframe[23].replace([r'.*Libertador+.*',r'.*(Sucre|Baruta|El Hatillo|Chacao|Altos|Guarenas|Valles|Barlovento).*',r'Ara.*',r'Apu.*',ur'Tác.*',r'Var.*',r'Mon.*',r'Por.*',r'Nue*.',r'Tru*.',r'Lar.*',r'Bol.*',r'Bar.*',r'Suc.*',r'Anz.*',ur'Mér.*',r'Delta.*',r'Yar.*',ur'Guár.*'],range(19),regex=True).astype('int')

#Limpiando la columna de residencia

output_dataframe.residency = dataframe[24].replace([r'.*Libertador+.*',r'.*Sucre',r'.*Baruta',r'.*El Hatillo',r'.*Chacao',r'.*Altos',r'.*Guarenas',r'.*Valles'],range(8),regex=True)
output_dataframe.residency = output_dataframe.residency.fillna(output_dataframe.residency.mode().iloc[0]).astype('int')

#Limpiando la columna con quien vive
output_dataframe.roomies = dataframe[25].replace([r'.*padres.*',r'.*(esposo(?<!su)|hijo).*',r'.*(mamá|madre).*',r'.*(papá|padre).*',r'.+maternos.*',r'.+paternos.*',r'sol[oa]',r'.+'], range(8), regex=True)

#Limpiando la columna tipo de vivienda
output_dataframe.houseType = dataframe[26].str.lower().replace([r'.*quinta.*',r'.*edific.*',r'.*urbano.*',r'.*rural.*',r'.*alquilada.*',r'.*vecindad.*',r'.*estudiantil.*'],range(7),regex=True)

#Limpiando la columna de beca
output_dataframe.scholarship = dataframe[34].apply(lambda x : 1500 if x <= 1500 else 2000).astype('float')

#Limpiando las columnas de los ingresos y egresos del Estudiante

dataframe.ix[:,35:48] = dataframe.ix[:,35:48].replace(np.nan,0)

output_dataframe.contributionHouseholder = dataframe[35]
output_dataframe.contributionFamily = dataframe[36]
output_dataframe.contributionActivities = dataframe[37]
output_dataframe.contributionMonthlyTotal = output_dataframe.scholarship + output_dataframe.contributionHouseholder + output_dataframe.contributionFamily + output_dataframe.contributionActivities
output_dataframe.foodExpenses = dataframe[39]
output_dataframe.transportExpenses = dataframe[40]
output_dataframe.medicalExpenses = dataframe[41]
output_dataframe.dentalExpenses = dataframe[42]
output_dataframe.personalExpenses = dataframe[43]
output_dataframe.rentExpenses = dataframe[44]
output_dataframe.studyMaterialExpenses = dataframe[45]
output_dataframe.recreationalExpenses = dataframe[46]
output_dataframe.otherExpenses = dataframe[47]
output_dataframe.totalExpenses = output_dataframe.foodExpenses + output_dataframe.transportExpenses + output_dataframe.medicalExpenses + output_dataframe.dentalExpenses + output_dataframe.personalExpenses + output_dataframe.rentExpenses + output_dataframe.studyMaterialExpenses + output_dataframe.recreationalExpenses + output_dataframe.otherExpenses

#Limpiando la columna responsable economico
output_dataframe.howHouseholder = dataframe[49].str.lower().replace([r'madre',r'padre',r'ambos.*',ur'cónyugue|esposo',r'.*(familiares|tia|hermano|hermana|abuela).*',r'usted.*|ninguno'],range(6),regex=True)

#El numero de cargafamiliar lo dejamos igual
output_dataframe.familyBurden = dataframe[50]

#Primer paso limpiando los ingresos/egresos del responsable ecconomico
dataframe.ix[:,51:63] = dataframe.ix[:,51:63].replace(np.nan,0)
#Segundo paso limpiando los ingresos/egresos del responsable economico
dataframe.ix[:,51:63] = dataframe.ix[:,51:63].replace([r'No|comodato|o'],0,regex=True)

for i in range(51, 64):
    dataframe[i] = dataframe[i].astype('string').str.replace(' ','').str.replace(',','.').str.replace('bs|\+','').apply(lambda x : x.replace(".","",1) if x.count('.') >= 2 else x)

#Copiando las columnas al dataframe resultante
output_dataframe.incomeHouseholder = dataframe[51].astype('float')
output_dataframe.incomeOther = dataframe[52].astype('float')
output_dataframe.incomeTotal = output_dataframe.incomeHouseholder + output_dataframe.incomeOther
output_dataframe.houseExpensesHouseholder = dataframe[54].astype('float')
output_dataframe.foodExpensesHouseholder = dataframe[55].astype('float')
output_dataframe.transportExpensesHouseholder = dataframe[56].astype('float')
output_dataframe.medicalExpensesHouseholder = dataframe[57].astype('float')
output_dataframe.dentalExpensesHouseholder = dataframe[58].astype('float')
output_dataframe.studyExpensesHouseholder = dataframe[59].astype('float')
output_dataframe.servicesExpensesHouseholder = dataframe[60].astype('float')
output_dataframe.condominiumExpensesHouseholder = dataframe[61].astype('float')
output_dataframe.otherExpensesHouseholder = dataframe[62].astype('float')
output_dataframe.totalExpensesHouseholder = output_dataframe.houseExpensesHouseholder + output_dataframe.foodExpensesHouseholder + output_dataframe.transportExpensesHouseholder + output_dataframe.medicalExpensesHouseholder + output_dataframe.dentalExpensesHouseholder + output_dataframe.studyExpensesHouseholder + output_dataframe.servicesExpensesHouseholder + output_dataframe.condominiumExpensesHouseholder + output_dataframe.otherExpensesHouseholder

#La columna de raiting la dejamos igual
output_dataframe.raiting = dataframe[64]

#Escribiendo en el archivo de salida (Vista Minable)
output_dataframe.to_csv(output_data_path, encoding='utf-8',index=False, float_format='%.3f', date_format='%d/%m/%Y')