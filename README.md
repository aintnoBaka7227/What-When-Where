# What When Where

This project contains a Python script, `data_generation.py`, that generates synthetic satellite observation data. It creates a random orbit, propagates that orbit forward in time, places a few synthetic observing sites on Earth, and writes the satellite state and line-of-sight pointing measurements to a text file.

## Dependencies

The project uses:

- `numpy` for vector and matrix math.
- `scipy` for 3D rotations.
- `numba` to compile the two-body propagator and RA/Dec conversion for speed.
- `astropy` to get Earth constants.

The dependencies are listed in `pyproject.toml`.

## Units

The script mixes constants from `astropy` with orbital calculations:

- `GM_EARTH` comes from `astropy.constants.GM_earth.value` in SI units, which is `m^3/s^2`.
- `R_EARTH` comes from `astropy.constants.R_earth.value` in meters.
- `mu = GM_EARTH / 1000000000`, converting Earth gravitational parameter from `m^3/s^2` to `km^3/s^2`.
- Satellite position values are handled in kilometers.
- Satellite velocity values are handled in kilometers per second.
- Observing site position values are stored in meters, then divided by `1000` when used with satellite positions.
- Angles are in radians.
- Time is in seconds.

## What The Script Does

When `data_generation (1).py` is run directly, it:

1. Selects a test case. The current file sets `test_case = 'a'`.
2. Randomly generates one set of classical orbital elements.
3. Converts those classical orbital elements into equinoctial orbital elements.
4. Converts the equinoctial elements into a GCRS-like position and velocity state.
5. Creates 3 observing sites.
6. For each site, generates 5 measurements over 5 seconds.
7. Propagates the satellite state to each measurement time using a two-body orbit model.
8. Converts the relative satellite-to-site position into right ascension and declination.
9. Appends each generated measurement row to `generated_data.txt`.

The current settings produce `3 * 5 = 15` rows per run.

Important: the script writes to `generated_data.txt`, but the repository currently contains a file named `generated_data (1).txt`. If you run the script as-is, it will create or append to `generated_data.txt`.

## Main Configuration

Inside the `if __name__ == "__main__":` block:

```python
test_case = 'a'
output_dir = 'generated_data.txt'
num_of_sites = 3
time_interval_between_sites = 120
num_of_pts_in_each_site = 5
measurement_time_in_each_site = 5
```

These values mean:

- `test_case`: chooses the random orbit family.
- `output_dir`: output text file path.
- `num_of_sites`: number of synthetic observing sites.
- `time_interval_between_sites`: time gap between site observation windows.
- `num_of_pts_in_each_site`: number of measurements generated at each site.
- `measurement_time_in_each_site`: length of each site observation window.

The script computes:

```python
measurement_time_interval_within_each_site =
    measurement_time_in_each_site / num_of_pts_in_each_site
```

With the current settings, measurements inside a site are taken 1 second apart.

## Test Cases

The test cases randomly choose different orbital ranges:

- `a`: perigee radius from 6880 to 8380 km, eccentricity from 0 to 0.01.
- `b`: perigee radius from 8380 to 9380 km, eccentricity from 0.01 to 0.2.
- `c`: perigee radius from 8380 to 9380 km, eccentricity from 0.2 to 0.4.
- `d`: perigee radius from 8380 to 9380 km, eccentricity from 0.4 to 0.6.

All test cases randomly choose:

- inclination from 1 to 179 degrees.
- right ascension of ascending node from 0 to `2*pi`.
- argument of perigee from 0 to `2*pi`.
- true anomaly from 0 to `2*pi`.

## Output Format

Each output row is comma-separated and has 12 columns:

```text
rx,ry,rz,vx,vy,vz,ra,dec,site_x,site_y,site_z,time
```

Column meanings:

- `rx`, `ry`, `rz`: satellite position in kilometers.
- `vx`, `vy`, `vz`: satellite velocity in kilometers per second.
- `ra`: right ascension of the satellite as seen from the synthetic site, in radians.
- `dec`: declination of the satellite as seen from the synthetic site, in radians.
- `site_x`, `site_y`, `site_z`: synthetic observing site position in meters.
- `time`: propagation time since the initial state, in seconds.

The file is opened in append mode, so running the script multiple times adds more rows instead of replacing the old data.

## Function Reference

### `gcrs_to_coe(gcrs)`

Converts a Cartesian position and velocity state into classical orbital elements.

Input:

- `gcrs[0:3]`: position vector.
- `gcrs[3:6]`: velocity vector.

Output:

```text
[rp, e, i, ohm, omega, v]
```

Where:

- `rp`: perigee radius.
- `e`: eccentricity.
- `i`: inclination.
- `ohm`: right ascension of the ascending node.
- `omega`: argument of perigee.
- `v`: true anomaly.

### `coe_to_gcrs(coe)`

Converts classical orbital elements into Cartesian position and velocity.

Input:

```text
[rp, e, i, ohm, omega, v]
```

The function computes semi-major axis `a`, semi-latus rectum `p`, position and velocity in the perifocal frame, then rotates them into the inertial frame.

Output:

```text
[rx, ry, rz, vx, vy, vz]
```

### `oe_to_gcrs(oe)`

Converts equinoctial orbital elements into Cartesian position and velocity.

Input:

```text
[a, ex, ey, hx, hy, l]
```

Where:

- `a`: semi-major axis.
- `ex`, `ey`: eccentricity vector components.
- `hx`, `hy`: inclination vector components.
- `l`: true longitude.

Output:

```text
[rx, ry, rz, vx, vy, vz]
```

This representation is often useful because it avoids some singularities that classical orbital elements have for circular or equatorial orbits.

### `gcrs_to_oe(gcrs)`

Converts Cartesian position and velocity into equinoctial orbital elements.

Output:

```text
[a, f, g, h, k, L]
```

The returned values correspond to semi-major axis, eccentricity components, inclination components, and longitude.

### `coe_to_eq_oe(coe)`

Converts classical orbital elements into equinoctial orbital elements.

Input:

```text
[rp, e, i, ohm, omega, v]
```

Output:

```text
[a, ex, ey, hx, hy, lv]
```

The script uses this function before creating the initial Cartesian state.

### `two_body(mu, tau, ri, vi)`

Propagates a satellite state forward by `tau` seconds using a two-body gravitational model.

Input:

- `mu`: Earth gravitational parameter in `km^3/s^2`.
- `tau`: propagation time in seconds.
- `ri`: initial position vector in kilometers.
- `vi`: initial velocity vector in kilometers per second.

Output:

- `rf`: final position vector.
- `vf`: final velocity vector.

The function is decorated with:

```python
@jit(nopython=True, nogil=True, cache=True)
```

That means Numba compiles it for better performance.

### `gcrs_to_radec(gcrs)`

Converts a Cartesian position vector into right ascension and declination.

Input:

- A 3D position vector, or a state where the first 3 values are position.

Output:

```text
[ra, dec]
```

The script uses this on the relative vector:

```python
satellite_position - observing_site_position
```

Because satellite positions are in kilometers and site positions are in meters, the site vector is divided by `1000` first.

### `radec_to_point(radec)`

Converts right ascension and declination into a unit direction vector.

Input:

```text
[ra, dec]
```

Output:

```text
[x, y, z]
```

This function is defined but not used by the current main data generation flow.

## Observing Site Generation

At the first measurement for each site, the script:

1. Takes the satellite radial direction.
2. Generates a random 3D rotation axis.
3. Generates a random rotation angle from 0 to 45 degrees.
4. Rotates the satellite radial direction by that amount.
5. Scales the result by Earth radius.

That produces a synthetic observing site on Earth near the satellite's ground-facing direction.

The same site is reused for all 5 measurements in that site's observation window.

## Notes And Caveats

- Output is appended, not overwritten.
- The random number generator is not seeded, so every run creates different data.
- The script uses `output_dir` as a filename, not a directory.
- `generated_data (1).txt` appears to be sample output, but the script writes to `generated_data.txt`.
- Some helper functions are not used in the current main flow, but they are useful for converting between orbital representations.
- The variable name `v` is used both for velocity vectors and true anomaly in different functions, so read each function in context.

## Running

Use:

```bash
python "data_generation (1).py"
```

This appends 15 generated rows to `generated_data.txt`.
