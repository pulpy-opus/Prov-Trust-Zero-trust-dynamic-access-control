# Permission threshold logging setup
import logging
import os
import json
import numpy as np
from tqdm import tqdm
# Statistical libraries
from scipy.stats import linregress

import pymannkendall as mk

artifact_dir = "./"

# IP trust value change logging setup
ip_trust_change_logger = logging.getLogger("ip_trust_change_logger")
ip_trust_change_logger.setLevel(logging.INFO)
ip_trust_change_log_path = artifact_dir + 'ip_trust_change.log'
# Clear log file contents
with open(ip_trust_change_log_path, 'w', encoding='utf-8'):
    pass
ip_trust_change_file_handler = logging.FileHandler(ip_trust_change_log_path)
ip_trust_change_file_handler.setLevel(logging.INFO)
ip_trust_change_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ip_trust_change_file_handler.setFormatter(ip_trust_change_formatter)
if not ip_trust_change_logger.hasHandlers():
    ip_trust_change_logger.addHandler(ip_trust_change_file_handler)

# Weights and parameters (adjustable)
DEFAULT_TRUST_FACTOR = 0.5  # Other trust factors
LAMBDA = 0.8  # Forgetting factor
BETA_INIT = 1.0  # Initial penalty factor
ALPHA = 0.9     # Penalty decay per anomalous event
QUEUE_LENGTH = 10  # Trust value queue length (each entry = 15 min)
W_BASIC = 0.2
W_ANOMALY = 0.8
DECLEARE_TRUST = 0.95  # Trust decay factor when no activity
REWARD_TRUST = 1.1  # Trust reward factor for normal activity
DECLEARE_RISK = 0.25  # Risk decay factor

# Initial permission threshold settings
THETA_HIGH = 0.8
THETA_LOW = 0.4

# --- Anomaly Score Computation ---
def compute_anomaly_score(loss_list, avg_loss_t, std_loss_t, count_t, total_anomalies_t,
                          prev_cum_risk=None, min_risk=0, max_risk=8,
                          alpha=0.7, beta=DECLEARE_RISK, scale_factor=1.0, k=0.5, m=2, eps=1e-9,
                          normal_count=0, ratio_gamma=1):
    # beta: risk decay coefficient
    # 1. Z-score normalization
    z_scores = [(loss_i - avg_loss_t) / (std_loss_t + eps) for loss_i in loss_list]
    # 2. Severity aggregation
    if z_scores:
        S_t = alpha * max(z_scores) + (1 - alpha) * np.mean(z_scores)
    else:
        S_t = 0
    cum_risk_t = S_t
    # # 3. Persistence (EWMA historical risk)
    # if prev_cum_risk is None:
    #     prev_cum_risk = S_t
    # cum_risk_t = beta * prev_cum_risk + (1 - beta) * S_t
    # # 4. Prevalence weight (Sigmoid compression)
    # W_p = scale_factor * (1 / (1 + np.exp(-k * (count_t - m))))
    # # 5. Final risk value
    raw_risk_t = cum_risk_t
    # raw_risk_t = cum_risk_t * W_p
    # Anomaly-to-normal behavior ratio
    anomaly_normal_ratio = count_t / (normal_count + count_t)
    # Normalize to 0-100
    risk_score_t = 100 * raw_risk_t * ratio_gamma * anomaly_normal_ratio  / max_risk
    # Weighted anomaly/normal ratio
    return risk_score_t, cum_risk_t, anomaly_normal_ratio

class IPBehaviorQueue:
    def __init__(self, ip, trust_basic=DEFAULT_TRUST_FACTOR):
        self.ip = ip
        self.trust_basic = trust_basic
        self.trust_queue = [(0, 0.5)]  # Initialize queue: timestamp=0, trust=0.5
        self.beta = BETA_INIT
        self.records = []
        self.trust_score = trust_basic
        self.theta_high = THETA_HIGH
        self.theta_low = THETA_LOW
    def add_record(self, time_window, loss_list, avg_loss, var_loss, count_t, total_anomalies_t, timestamp, min_risk=0, max_risk=10, normal_count=0):
        std_loss = np.sqrt(var_loss)
        anomaly_score, cum_risk, anomaly_normal_ratio = compute_anomaly_score(
            loss_list, avg_loss, std_loss, count_t, total_anomalies_t,
            prev_cum_risk=self.records[-1]['cum_risk'] if self.records else None,
            min_risk=min_risk, max_risk=max_risk,
            normal_count=normal_count)
        trust_t = self.calculate_trust_t(anomaly_score / 100)  # Normalize to [0, 1]
        self.trust_queue.append((timestamp, trust_t))
        if len(self.trust_queue) > QUEUE_LENGTH:
            self.trust_queue.pop(0)
        self.records.append({
            'time_window': time_window,
            'loss_list': loss_list,
            'avg_loss': avg_loss,
            'var_loss': var_loss,
            'count_t': count_t,
            'total_anomalies_t': total_anomalies_t,
            'anomaly_score': anomaly_score,
            'cum_risk': cum_risk,
            'trust_t': trust_t,
            'timestamp': timestamp,
            'normal_count': normal_count,
            'anomaly_normal_ratio': anomaly_normal_ratio
        })
        if anomaly_score > 80:
            self.beta *= ALPHA

    def calculate_trust_t(self, anomaly_score):
        # Use the latest trust value in queue as base
        base_trust = self.trust_queue[-1][1] if self.trust_queue else 0.5
        # Change factor based on normalized anomaly score
        # if anomaly_score < 0.08:
        #     change_factor = 1 + anomaly_score
        # else:
        #     change_factor = (1 - anomaly_score) ** 2
        change_factor = (1 - anomaly_score) ** 2
        # New trust value
        new_trust = min(base_trust * change_factor, 1.0)
        return new_trust

    def calculate_final_trust(self, current_time):
        # Rolling window weighted average + penalty factor
        numerator = 0
        denominator = 0
        for timestamp, trust_i in self.trust_queue:
            weight = LAMBDA ** (current_time - timestamp)
            numerator += weight * trust_i
            denominator += weight
        if denominator == 0:
            return self.trust_basic
        return numerator / (denominator * self.beta)



def calculate_permission_threshold(ip_trust_dict, trust_history):
    # Dynamic permission threshold adjustment (Mann-Kendall trend test + linear regression slope)
    # trust_history: {ip: [trust1, trust2, ...]}
    all_trusts = []
    for ip in trust_history:
        all_trusts.extend(trust_history[ip])
    if not all_trusts or len(all_trusts) < 3:
        # Too few samples, return default thresholds
        return THETA_HIGH, THETA_LOW
    x = np.arange(len(all_trusts))
    y = np.array(all_trusts)
    # Linear regression slope
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # Mann-Kendall trend test
    if mk:
        mk_result = mk.original_test(y)
        # In newer pymannkendall, S is test_statistic
        S = getattr(mk_result, 'test_statistic', None)
        P = mk_result.p
        trend = mk_result.trend
    theta_high, theta_low = THETA_HIGH, THETA_LOW
    # Permission threshold adjustment logic
    if trend == 'increasing' and P < 0.05 and slope > 0:
        theta_high = max(THETA_HIGH - 0.05, 0.6)
    elif trend == 'decreasing' and P < 0.05 and slope < 0:
        theta_low = min(THETA_LOW + 0.05, 0.6)
    return theta_high, theta_low

def main(artifact_dir):
    iplog_dir = os.path.join(artifact_dir, "ip_behavior_logs")
    netflow_dir = os.path.join(artifact_dir, "netflow_ip_logs")
    ip_trust_dict = {}
    trust_history = {}

    time_windows = sorted([f for f in os.listdir(netflow_dir) if f.startswith("netflow_ip_log_")])
    all_ips = set()
    for idx, netflow_fname in enumerate(tqdm(time_windows, desc="Processing time windows")):
        time_window = netflow_fname.replace("netflow_ip_log_", "").replace(".json", "")
        netflow_path = os.path.join(netflow_dir, netflow_fname)
        with open(netflow_path, "r") as f:
            netflow_ip_log = json.load(f)
        all_ips.update(netflow_ip_log.keys())
        # Initialize new IP
        for ip in netflow_ip_log:
            if ip not in ip_trust_dict:
                trust_basic = DEFAULT_TRUST_FACTOR
                ip_trust_dict[ip] = IPBehaviorQueue(ip, trust_basic)
                trust_history[ip] = []
        abnormal_fname = f"abnormal_ip_log_{time_window}.json"
        # abnormal_path = os.path.join(iplog_dir, abnormal_fname)
        abnormal_path = os.path.join(netflow_dir, abnormal_fname)
        total_anomalies_t = 0
        ip_loss_dict = {}
        ip_normal_count_dict = {}
        abnormal_ips = set()
        if os.path.exists(abnormal_path):
            with open(abnormal_path, "r") as f:
                abnormal_ip_log = json.load(f)
            for ip, info in abnormal_ip_log.items():
                loss_list = [entry.get('loss', 0) for entry in info.get('entries', [])]
                total_anomalies_t += len(loss_list)
                ip_loss_dict[ip] = loss_list
                # Count normal_entries
                normal_count = len(info.get('normal_entries', []))
                ip_normal_count_dict[ip] = normal_count
            for ip, info in abnormal_ip_log.items():
                loss_list = ip_loss_dict[ip]
                avg_loss = np.mean(loss_list) if loss_list else 0
                var_loss = np.var(loss_list) if loss_list else 0
                count_t = len(loss_list)
                normal_count = ip_normal_count_dict.get(ip, 0)
                timestamp = idx
                min_risk, max_risk = 0, 10
                ip_trust_dict[ip].add_record(
                    time_window, loss_list, avg_loss, var_loss, count_t, total_anomalies_t, timestamp,
                    min_risk=min_risk, max_risk=max_risk, normal_count=normal_count)
                trust = ip_trust_dict[ip].calculate_final_trust(timestamp)
                trust_history[ip].append(trust)
                abnormal_ips.add(ip)
        # Per-IP dynamic permission threshold adjustment (Mann-Kendall + linear regression)
        for ip in all_ips:
            theta_high, theta_low = calculate_permission_threshold(ip_trust_dict, {ip: trust_history[ip]})
            ip_trust_dict[ip].theta_high = theta_high
            ip_trust_dict[ip].theta_low = theta_low
    # Permission threshold logging disabled
        # Trust reward/decay logic
        operated_ips = set(netflow_ip_log.keys())
        for ip in all_ips:
            if ip in abnormal_ips:
                continue  # Already processed
            elif ip in operated_ips:
                # Activity with no anomaly: reward
                last_trust = ip_trust_dict[ip].trust_queue[-1][1] if ip_trust_dict[ip].trust_queue else ip_trust_dict[ip].trust_basic
                reward_trust = min(last_trust * REWARD_TRUST, 1.0)
                # Also record trust value to trust_queue and records on reward
                ip_trust_dict[ip].trust_queue.append((idx, reward_trust))
                if len(ip_trust_dict[ip].trust_queue) > QUEUE_LENGTH:
                    ip_trust_dict[ip].trust_queue.pop(0)
                ip_trust_dict[ip].records.append({
                    'time_window': time_window,
                    'loss_list': [],
                    'avg_loss': 0,
                    'var_loss': 0,
                    'count_t': 0,
                    'total_anomalies_t': 0,
                    'anomaly_score': 0,
                    'cum_risk': 0,
                    'trust_t': reward_trust,
                    'timestamp': idx,
                    'normal_count': 0,
                    'anomaly_normal_ratio': 0
                })
                trust = ip_trust_dict[ip].calculate_final_trust(idx)
                trust_history[ip].append(trust)
            else:
                # No activity, no anomaly: decay
                last_trust = ip_trust_dict[ip].trust_queue[-1][1] if ip_trust_dict[ip].trust_queue else ip_trust_dict[ip].trust_basic
                decay_trust = max(last_trust * DECLEARE_TRUST, 0.0)
                ip_trust_dict[ip].trust_queue.append((idx, decay_trust))
                if len(ip_trust_dict[ip].trust_queue) > QUEUE_LENGTH:
                    ip_trust_dict[ip].trust_queue.pop(0)
                ip_trust_dict[ip].records.append({
                    'time_window': time_window,
                    'loss_list': [],
                    'avg_loss': 0,
                    'var_loss': 0,
                    'count_t': 0,
                    'total_anomalies_t': 0,
                    'anomaly_score': 0,
                    'cum_risk': 0,
                    'trust_t': decay_trust,
                    'timestamp': idx,
                    'normal_count': 0,
                    'anomaly_normal_ratio': 0
                })
                trust = ip_trust_dict[ip].calculate_final_trust(idx)
                trust_history[ip].append(trust)
        # Log output
        ip_log_record = {
            'time_window': time_window,
            'ip_stats': []
        }
        for ip in all_ips:
            record = ip_trust_dict[ip].records[-1] if ip_trust_dict[ip].records else {}
            risk = record.get('anomaly_score', 0)
            trust_val = trust_history[ip][-1] if trust_history[ip] else ip_trust_dict[ip].trust_basic
            anomaly_normal_ratio = record.get('anomaly_normal_ratio', 0)
            ip_log_record['ip_stats'].append({
                'ip': ip,
                'risk_score': risk,
                'trust_value': trust_val,
                'theta_high': ip_trust_dict[ip].theta_high,
                'theta_low': ip_trust_dict[ip].theta_low,
                'anomaly_normal_ratio': anomaly_normal_ratio
            })
        ip_trust_change_logger.info(json.dumps(ip_log_record, ensure_ascii=False))

if __name__ == "__main__":
    main("./artifact")