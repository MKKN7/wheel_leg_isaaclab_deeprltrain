"""Angle-only authorized high-speed screening; preserve strict evidence."""
import argparse
import json
from pathlib import Path


def analyze(directory):
    source = json.loads((directory / 'strict_screen.json').read_text())
    rows = []
    for original in source['rows']:
        row = dict(original)
        checks = dict(row['checks'])
        relaxed = row['vx'] >= 1.6 and abs(row['yaw']) <= 1e-6
        if relaxed:
            angle = row['angle_p95_deg']
            checks['straight_angles'] = angle is not None and max(angle) <= 3.0
        row.update(engineering_checks=checks,
                   engineering_screen_pass=all(checks.values()),
                   angle_relaxation_applied=relaxed)
        rows.append(row)
    summary = []
    for point in source['summary']:
        selected = [x for x in rows if x['vx'] == point['vx'] and x['yaw'] == point['yaw']]
        summary.append(dict(point, engineering_screen_pass=sum(x['engineering_screen_pass'] for x in selected)))
    result = dict(summary=summary, rows=rows, promoted=False,
                  scope='User-authorized straight vx>=1.6 angle P95<=3deg; other checks unchanged. Oscillation, drift, multi-seed acceptance still separate. Strict source retained.')
    (directory / 'highspeed_engineering_screen.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    print(json.dumps(analyze(parser.parse_args().directory)['summary'], indent=2))
