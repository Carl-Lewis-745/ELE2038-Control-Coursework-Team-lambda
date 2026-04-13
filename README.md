# Control Systems Coursework – PID Control

## Overview

This project models and controls a nonlinear electromechanical system (ball on an inclined plane with electromagnetic actuation).

A PID controller is designed using a linearised model and validated using nonlinear simulation.

---

## What’s included

The code generates:

* Step response (linear vs nonlinear)
* Disturbance rejection
* Saturation effects with anti-windup
* Frequency-domain plots (L, S, T)
* Monte Carlo robustness test

---

## Key objective

Design a controller that achieves:

* Stability
* Zero steady-state error
* Good damping
* Disturbance rejection

---

## How to run

Install dependencies:

```bash
pip install numpy scipy matplotlib control
```

Run:

```bash
python main.py
```

---

## Notes

* Based on linearisation around ( x = 0.5 , \text{m} )
* Includes nonlinear validation and robustness analysis
* Developed for ELE2038 Signals and Control coursework
