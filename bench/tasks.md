# NeuroTiny Benchmark Tasks

## Task: Simple Product Query
- Query: Get the price of AirPods Pro
- Schema: product_v1
- Expected: Product with name, price, and availability

## Task: Product with Web Fetch
- Query: Extract laptop details from online store
- URL: https://shop.example.com/laptops/dell-xps-15
- Selector: .product-container
- Schema: product_v1
- Expected: Product with full details from webpage

## Task: Blog Post Extraction
- Query: Get the latest blog post about artificial intelligence
- Schema: post_v1
- Expected: Post with title, author, date, and tags

## Task: Blog with URL
- Query: Extract blog post from website
- URL: https://blog.example.com/posts/ai-revolution
- Selector: article.post
- Schema: post_v1
- Expected: Complete blog post with metadata

## Task: Event Information
- Query: Conference on March 20, 2024 at Moscone Center
- Schema: event_v1
- Expected: Event with date, time, and location

## Task: Event with Web Scraping
- Query: Get event details from webpage
- URL: https://events.example.com/tech-summit-2024
- Selector: .event-details
- Schema: event_v1
- Expected: Full event information

## Task: Multiple Products
- Query: Find all smartphones under $500
- Schema: product_v1
- Expected: Product matching price constraint

## Task: Author Posts
- Query: All posts by John Smith about machine learning
- Schema: post_v1
- Expected: Post with specific author and topic

## Task: Upcoming Events
- Query: Tech events in San Francisco next month
- Schema: event_v1
- Expected: Event with location and future date

## Task: Product Comparison
- Query: Compare iPhone 15 and Samsung Galaxy S24 prices
- Schema: product_v1
- Expected: Product with comparison data

## Task: Blog Tags
- Query: Posts tagged with Python and Data Science
- Schema: post_v1
- Expected: Post with multiple tags

## Task: Event Series
- Query: All workshops at Developer Conference 2024
- Schema: event_v1
- Expected: Event with workshop type

## Task: Product Stock Check
- Query: Check if PlayStation 5 is in stock
- Schema: product_v1
- Expected: Product with stock status

## Task: Recent Posts
- Query: Blog posts from last week about cloud computing
- Schema: post_v1
- Expected: Recent post with date and topic

## Task: Virtual Event
- Query: Online webinar about AI ethics on Friday 2PM
- Schema: event_v1
- Expected: Virtual event with time

## Task: Product with Reviews
- Query: Get product info for highly rated headphones
- URL: https://reviews.example.com/audio/best-headphones
- Selector: .product-review
- Schema: product_v1
- Expected: Product with rating information

## Task: Tutorial Post
- Query: Python tutorial for beginners with code examples
- Schema: post_v1
- Expected: Tutorial post with educational content

## Task: Conference Schedule
- Query: Day 1 schedule for Tech Summit
- URL: https://summit.example.com/schedule/day1
- Selector: .schedule-item
- Schema: event_v1
- Expected: Multiple events from schedule

## Task: Sale Products
- Query: Black Friday deals on electronics
- Schema: product_v1
- Expected: Products with sale prices

## Task: Guest Posts
- Query: Guest author posts on the company blog
- Schema: post_v1
- Expected: Posts by external authors

## Task: Product Bundle
- Query: Gaming PC bundle with monitor and accessories
- Schema: product_v1
- Expected: Bundle product with components

## Task: Event Registration
- Query: Free workshop on web development with registration link
- Schema: event_v1
- Expected: Event with registration details

## Task: Product Specifications
- Query: Technical specs for MacBook Pro M3
- URL: https://specs.example.com/apple/macbook-pro-m3
- Selector: .specs-table
- Schema: product_v1
- Expected: Product with detailed specifications

## Task: Blog Series
- Query: Part 3 of the microservices architecture series
- Schema: post_v1
- Expected: Series post with part number

## Task: Hybrid Event
- Query: AI conference with both in-person and virtual attendance
- Schema: event_v1
- Expected: Hybrid event with dual format