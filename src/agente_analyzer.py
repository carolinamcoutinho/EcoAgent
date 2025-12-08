import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class EnergyAnalysisAgent:
    """
    Agente inteligente para análise de consumo energético.
    Identifica os principais fatores de influência e gera explicações contextualizadas.
    """
    
    def __init__(self):
        self.knowledge_base = {
            'Temperature': {
                'high': {
                    'threshold': 28,
                    'message': "A **temperatura elevada** ({value}°C) está aumentando significativamente o consumo. "
                              "Em dias quentes, sistemas de climatização trabalham mais para resfriar o ambiente. "
                              "💡 **Dica**: Considere usar ventiladores, fechar cortinas durante o dia e programar o "
                              "ar-condicionado para temperaturas mais econômicas (24-26°C)."
                },
                'low': {
                    'threshold': 18,
                    'message': "A **temperatura baixa** ({value}°C) está impactando o consumo energético. "
                              "Aquecedores e sistemas de climatização consomem energia para aquecer o ambiente. "
                              "💡 **Dica**: Use roupas adequadas, isole janelas e portas, e ajuste o termostato para "
                              "temperaturas moderadas (20-22°C)."
                }
            },
            'Humidity': {
                'high': {
                    'threshold': 70,
                    'message': "A **umidade alta** ({value}%) está contribuindo para o maior consumo. "
                              "Ambientes úmidos dificultam o resfriamento e podem exigir mais do sistema de climatização. "
                              "💡 **Dica**: Use desumidificadores apenas quando necessário e mantenha boa ventilação natural."
                },
                'low': {
                    'threshold': 30,
                    'message': "A **umidade baixa** ({value}%) pode estar afetando o conforto térmico. "
                              "Ar muito seco pode fazer com que você ajuste o termostato desnecessariamente. "
                              "💡 **Dica**: Considere usar umidificadores eficientes ou recipientes com água."
                }
            },
            'SquareFootage': {
                'high': {
                    'threshold': 2000,
                    'message': "O **tamanho do ambiente** ({value} ft²) é um fator importante no consumo. "
                              "Espaços maiores naturalmente demandam mais energia para climatização e iluminação. "
                              "💡 **Dica**: Considere climatizar apenas as áreas em uso, usar ventilação natural e "
                              "investir em iluminação LED eficiente."
                }
            },
            'Occupancy': {
                'high': {
                    'threshold': 4,
                    'message': "A **alta ocupação** ({value} pessoas) está elevando o consumo energético. "
                              "Mais pessoas geram calor corporal e geralmente aumentam o uso de equipamentos. "
                              "💡 **Dica**: Eduque os ocupantes sobre economia de energia, desligue equipamentos não utilizados "
                              "e otimize o uso da climatização."
                }
            },
            'HVACUsage': {
                'on': {
                    'message': "O **sistema de climatização (HVAC) ligado** é o principal consumidor de energia. "
                              "Sistemas HVAC podem representar 40-60% do consumo total de um ambiente. "
                              "💡 **Dica**: Ajuste para temperaturas econômicas, faça manutenção regular dos filtros, "
                              "use programação inteligente e considere desligar quando o ambiente estiver vazio."
                }
            },
            'LightingUsage': {
                'on': {
                    'message': "A **iluminação ligada** está contribuindo para o consumo atual. "
                              "Embora menos que climatização, a iluminação pode representar 10-20% do consumo. "
                              "💡 **Dica**: Substitua lâmpadas por LEDs, use luz natural sempre que possível e "
                              "instale sensores de presença em áreas de circulação."
                }
            },
            'RenewableEnergy': {
                'low': {
                    'threshold': 30,
                    'message': "O **baixo uso de energia renovável** ({value}%) significa maior dependência da rede elétrica. "
                              "Aumentar o uso de fontes renováveis pode reduzir custos e impacto ambiental. "
                              "💡 **Dica**: Considere instalar painéis solares, participar de programas de energia limpa "
                              "ou investir em microgeração distribuída."
                }
            },
            'Holiday': {
                'yes': {
                    'message': "É um **dia de feriado**, quando padrões de consumo geralmente são diferentes. "
                              "Feriados podem ter mais pessoas em casa por mais tempo. "
                              "💡 **Dica**: Aproveite para revisar hábitos de consumo e planejar o uso de equipamentos."
                }
            }
        }
        
        # Dicionário de dias da semana em português
        self.day_names = {
            'Day_Monday': 'Segunda-feira',
            'Day_Tuesday': 'Terça-feira',
            'Day_Wednesday': 'Quarta-feira',
            'Day_Thursday': 'Quinta-feira',
            'Day_Friday': 'Sexta-feira',
            'Day_Saturday': 'Sábado',
            'Day_Sunday': 'Domingo'
        }

    def analyze_consumption(self, 
                          input_data: Dict, 
                          coefficients: pd.Series, 
                          prediction: float,
                          top_n: int = 3) -> Dict:
        """
        Analisa o consumo energético e gera insights inteligentes.
        
        Args:
            input_data: Dados de entrada do usuário
            coefficients: Coeficientes do modelo (importância das features)
            prediction: Valor previsto de consumo
            top_n: Número de fatores principais a analisar
            
        Returns:
            Dicionário com análise completa
        """
        top_factors = coefficients.abs().nlargest(top_n)
        
        explanations = []
        
        for feature_name, coef_value in top_factors.items():
            explanation = self._generate_explanation(
                feature_name, 
                coef_value, 
                input_data
            )
            if explanation:
                explanations.append(explanation)
        
        summary = self._generate_summary(prediction, input_data, top_factors)
        
        consumption_level = self._classify_consumption(prediction)
        
        return {
            'summary': summary,
            'consumption_level': consumption_level,
            'top_factors': top_factors.to_dict(),
            'explanations': explanations,
            'recommendations': self._generate_recommendations(input_data, top_factors)
        }
    
    def _generate_explanation(self, 
                            feature_name: str, 
                            coefficient: float, 
                            input_data: Dict) -> Dict:
        """Gera explicação contextualizada para um fator específico."""
        
        clean_feature = feature_name.replace('Dia ', 'Day_')
        
        explanation = {
            'feature': feature_name,
            'coefficient': coefficient,
            'impact': 'positivo' if coefficient > 0 else 'negativo',
            'message': ''
        }
        
        if clean_feature == 'Temperature':
            temp_value = input_data.get('Temperature', 0)
            if temp_value >= self.knowledge_base['Temperature']['high']['threshold']:
                explanation['message'] = self.knowledge_base['Temperature']['high']['message'].format(value=temp_value)
            elif temp_value <= self.knowledge_base['Temperature']['low']['threshold']:
                explanation['message'] = self.knowledge_base['Temperature']['low']['message'].format(value=temp_value)
            else:
                explanation['message'] = f"A temperatura ({temp_value}°C) está em uma faixa moderada."
        
        elif clean_feature == 'Humidity':
            humidity_value = input_data.get('Humidity', 0)
            if humidity_value >= self.knowledge_base['Humidity']['high']['threshold']:
                explanation['message'] = self.knowledge_base['Humidity']['high']['message'].format(value=humidity_value)
            elif humidity_value <= self.knowledge_base['Humidity']['low']['threshold']:
                explanation['message'] = self.knowledge_base['Humidity']['low']['message'].format(value=humidity_value)
        
        elif clean_feature == 'SquareFootage':
            area_value = input_data.get('SquareFootage', 0)
            if area_value >= self.knowledge_base['SquareFootage']['high']['threshold']:
                explanation['message'] = self.knowledge_base['SquareFootage']['high']['message'].format(value=area_value)
        
        elif clean_feature == 'Occupancy':
            occupancy_value = input_data.get('Occupancy', 0)
            if occupancy_value >= self.knowledge_base['Occupancy']['high']['threshold']:
                explanation['message'] = self.knowledge_base['Occupancy']['high']['message'].format(value=occupancy_value)
        
        elif clean_feature == 'HVACUsage':
            if input_data.get('HVACUsage', 0) == 1:
                explanation['message'] = self.knowledge_base['HVACUsage']['on']['message']
        
        elif clean_feature == 'LightingUsage':
            if input_data.get('LightingUsage', 0) == 1:
                explanation['message'] = self.knowledge_base['LightingUsage']['on']['message']
        
        elif clean_feature == 'RenewableEnergy':
            renewable_value = input_data.get('RenewableEnergy', 0)
            if renewable_value <= self.knowledge_base['RenewableEnergy']['low']['threshold']:
                explanation['message'] = self.knowledge_base['RenewableEnergy']['low']['message'].format(value=renewable_value)
        
        elif clean_feature == 'Holiday':
            if input_data.get('Holiday', 0) == 1:
                explanation['message'] = self.knowledge_base['Holiday']['yes']['message']
        
        elif clean_feature.startswith('Day_'):
            day_name = self.day_names.get(clean_feature, clean_feature)
            if input_data.get(clean_feature, 0) == 1:
                explanation['message'] = f"O dia da semana (**{day_name}**) influencia os padrões de consumo devido a rotinas específicas."
        
        return explanation if explanation['message'] else None
    
    def _generate_summary(self, 
                         prediction: float, 
                         input_data: Dict, 
                         top_factors: pd.Series) -> str:
        """Gera um resumo executivo da análise."""
        
        consumption_level = self._classify_consumption(prediction)
        
        main_factor = top_factors.index[0]
        main_factor_clean = main_factor.replace('Dia ', '')
        
        factor_translations = {
            'Temperature': 'temperatura',
            'Humidity': 'umidade',
            'SquareFootage': 'tamanho do ambiente',
            'Occupancy': 'ocupação',
            'HVACUsage': 'sistema de climatização',
            'LightingUsage': 'iluminação',
            'RenewableEnergy': 'energia renovável',
            'Holiday': 'feriado'
        }
        
        main_factor_pt = factor_translations.get(main_factor_clean, main_factor_clean)
        
        if main_factor_clean.startswith('Day_'):
            main_factor_pt = self.day_names.get(main_factor_clean, main_factor_clean)
        
        summary = f"""
### 🎯 Resumo da Análise

Seu consumo estimado é de **{prediction:.2f} kWh**, classificado como **{consumption_level['label']}**.

O fator que mais influencia seu consumo atual é: **{main_factor_pt}**.

{consumption_level['description']}
        """
        
        return summary.strip()
    
    def _classify_consumption(self, prediction: float) -> Dict:
        """Classifica o nível de consumo."""
        
        if prediction < 50:
            return {
                'level': 'low',
                'label': 'Baixo ⚡',
                'description': 'Parabéns! Seu consumo está em um nível eficiente. Continue com boas práticas de economia.',
                'color': 'green'
            }
        elif prediction < 76:
            return {
                'level': 'medium',
                'label': 'Moderado ⚡⚡',
                'description': 'Seu consumo está em um nível razoável, mas há oportunidades de otimização.',
                'color': 'orange'
            }
        else:
            return {
                'level': 'high',
                'label': 'Alto ⚡⚡⚡',
                'description': 'Seu consumo está elevado. Recomendamos atenção especial às dicas de economia.',
                'color': 'red'
            }
    
    def _generate_recommendations(self, 
                                 input_data: Dict, 
                                 top_factors: pd.Series) -> List[str]:
        """Gera recomendações priorizadas."""
        
        recommendations = []
        
        for feature_name in top_factors.index[:3]:
            clean_feature = feature_name.replace('Dia ', 'Day_')
            
            if clean_feature == 'HVACUsage' and input_data.get('HVACUsage', 0) == 1:
                recommendations.append("🌡️ Ajuste o termostato para temperaturas econômicas (24-26°C no verão, 20-22°C no inverno)")
                recommendations.append("🔧 Faça manutenção preventiva do sistema HVAC a cada 6 meses")
            
            elif clean_feature == 'Temperature':
                temp = input_data.get('Temperature', 0)
                if temp > 28:
                    recommendations.append("☀️ Use cortinas ou persianas para bloquear o calor solar direto")
                elif temp < 18:
                    recommendations.append("🧥 Vista-se adequadamente para reduzir a necessidade de aquecimento")
            
            elif clean_feature == 'SquareFootage':
                recommendations.append("🏠 Climatize apenas os ambientes ocupados, não toda a casa")
            
            elif clean_feature == 'LightingUsage' and input_data.get('LightingUsage', 0) == 1:
                recommendations.append("💡 Substitua lâmpadas convencionais por LEDs (economia de até 80%)")
            
            elif clean_feature == 'RenewableEnergy':
                if input_data.get('RenewableEnergy', 0) < 30:
                    recommendations.append("🌞 Considere investir em painéis solares ou energia renovável")
        
        recommendations.append("📊 Monitore seu consumo regularmente para identificar padrões")
        recommendations.append("⏰ Use temporizadores e automação para otimizar o uso de equipamentos")
        
        return recommendations[:5] 