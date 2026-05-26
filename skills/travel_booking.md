# Travel booking playbook

Use this playbook for hotel / travel-site tasks such as Expedia, Booking.com, Agoda, or checkout-flow exploration.

1. **Start in browser** — Prefer `browser_playwright` for dynamic travel sites. Begin with `navigate`, then `debug_snapshot` or `screenshot` so you can see the current state before interacting.
2. **Keep live preview visible** — If the task mentions live preview, start or confirm the relevant local app server, call `live_preview` with `probe`, then `set_active_port` so the Browser tab can follow progress.
3. **Work step-by-step** — For forms, fill one logical group at a time: destination, dates, guests, filters, then submit. After each major interaction, verify the page changed as expected before continuing.
4. **Prefer safe exploration** — If the user is testing capability, use sample search inputs unless they provided exact trip details. Clearly label sample data as sample data.
5. **Compare options explicitly** — When results load, extract enough detail to compare `cheapest`, `best rated`, and `best value` options. Mention trade-offs like price, rating, review count, and refund policy when visible.
6. **Pause on sensitive steps** — Stop and report when you hit login, captcha, OTP, identity verification, payment entry, or final confirmation. Never submit payment details or confirm a real booking without explicit user approval.
7. **Handle blockers carefully** — If selectors are unstable or the layout changes, take a fresh snapshot/screenshot, re-evaluate the page, and try a more robust locator before declaring failure.
8. **Report progress** — Provide short updates after each major milestone: search form found, search submitted, results compared, property selected, checkout reached, blocked, or awaiting approval.
