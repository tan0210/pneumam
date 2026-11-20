---
title: Samplers and guidance
description: Choose between the Euler and ER-SDE samplers, and set the guidance scale.
sidebar:
  order: 3
---

## Samplers

Pneumam has two samplers. Both work in ComfyUI, on the command line and in Python.

| Sampler | How it behaves |
|---|---|
| `euler` | It gives the same image every time for the same seed and settings. Start with this one. |
| `er_sde` | It adds fresh noise at each step, so its images differ from `euler`'s for the same seed. |

`er_sde` takes three extra settings. `max_stage` sets the solver stage, from 1 to 3. `s_noise` scales the fresh noise, and `noise_scale` chooses between `"er"` and `"ode"` noise.

## Guidance

The guidance scale sets how closely the image follows the prompt. Around 4 works well for Anima, and it is the default everywhere. A scale of 1 turns guidance off, which makes each step about twice as fast.

## Steps and shift

30 steps is the default and works well for most images. `shift` moves the steps toward rough shapes or toward fine detail. In ComfyUI, use the `simple` scheduler.

## Seeds

The same seed gives the same starting noise. Pneumam's KSampler starts from the same noise as ComfyUI's stock KSampler, so a seed you liked there gives the same picture here.
