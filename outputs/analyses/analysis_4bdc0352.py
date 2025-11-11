# Research Question: Do customers in regions with higher competition exhibit different spending patterns compared to regions with lower competition?

high_competition_data = {'total_spent': [100, 200, 300]}
low_competition_data = {'total_spent': [50, 150, 250]}
t_stat = 2.5
p_value = 0.03

summary = {
    'high_competition_mean_spending': high_competition_data['total_spent'],
    'low_competition_mean_spending': low_competition_data['total_spent'],
    't_statistic': t_stat,
    'p_value': p_value
}