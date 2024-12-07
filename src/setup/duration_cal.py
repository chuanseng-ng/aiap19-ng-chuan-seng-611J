import time


# Calculate runtime duration with its tag
def duration_cal(duration: float):
    if duration > 60:
        if duration > 3600:
            duration = duration / 3600
            tag = "hr"
        else:
            duration = duration / 60
            tag = "min"
    else:
        tag = "sec"

    return duration, tag


# Print out runtime and return current time, takes in time value to offset
def duration_print(input_time: float, step_cnt: int):
    current_time = time.time()
    current_duration, current_tag = duration_cal(current_time - input_time)
    print(f"Part {step_cnt} has run for {current_duration:.3f} {current_tag}!")
    print()
    step_cnt = step_cnt + 1

    return current_time, step_cnt
