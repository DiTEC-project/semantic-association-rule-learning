from statistics import mean


def get_coverage_false_positives(ground_truth, test):
    """
    coverage and false positives of AE SemRL according to the ground truth (FP-Growth)
    """
    if len(ground_truth) == 0:
        return 1, len(test)
    covered_rule_count = 0
    not_covered_rules = []
    for actual_rule_index in range(len(ground_truth)):
        actual_rule = ground_truth[actual_rule_index]
        found = False
        for test_rule_index in range(len(test)):
            test_rule = test[test_rule_index]
            if len(actual_rule['antecedents']) != len(test_rule['antecedents']):
                continue
            antecedent_matched = True
            for antecedent1 in test_rule['antecedents']:
                if antecedent1 not in actual_rule['antecedents']:
                    antecedent_matched = False
                    break
            if antecedent_matched:
                if test_rule['consequent'] == actual_rule['consequent']:
                    if actual_rule['consequent_index'] < len(actual_rule['antecedents']) and \
                            test_rule['consequent_index'] < len(test_rule['antecedents']):
                        if test_rule['antecedents'][test_rule['consequent_index']] == \
                                actual_rule['antecedents'][actual_rule['consequent_index']]:
                            covered_rule_count += 1
                            found = True
                            break
                    elif actual_rule['consequent_index'] >= len(actual_rule['antecedents']) and \
                            test_rule['consequent_index'] >= len(test_rule['antecedents']):
                        covered_rule_count += 1
                        found = True
                        break
        if not found:
            not_covered_rules.append(actual_rule)

    false_positive_count = len(test) - covered_rule_count
    return covered_rule_count / len(ground_truth), false_positive_count


def evaluate_rules(association_rules):
    """
    average support, confidence, lift, conviction, leverage and zhangs_metric value of the rules found by AE SemRL
    """
    support_list = []
    confidence_list = []
    lift_list = []
    leverage_list = []
    # conviction_list = []
    zhangs_metric_list = []
    for rule in association_rules:
        support_list.append(rule['support'])
        confidence_list.append(rule['confidence'])
        lift_list.append(rule['lift'])
        leverage_list.append(rule['leverage'])
        zhangs_metric_list.append(rule['zhangs_metric'])

    return mean(support_list), mean(confidence_list), mean(lift_list), mean(leverage_list), mean(zhangs_metric_list)
