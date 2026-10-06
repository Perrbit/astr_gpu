#pragma once

// Standalone dependency probes do not link the solver's native scheduler.
extern "C" int astr_insitu_product_clocks_active() __attribute__((weak));
extern "C" int astr_insitu_product_scene_due(const char*) __attribute__((weak));
inline bool astr_insitu_independent_products() {
  return astr_insitu_product_clocks_active && astr_insitu_product_clocks_active();
}
inline bool astr_insitu_scene_due(const char* name) {
  return !astr_insitu_independent_products() || astr_insitu_product_scene_due(name);
}
