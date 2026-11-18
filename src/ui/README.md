# src/ui (owner: Prithika)

React + Vite + Leaflet evaluator page. It only talks to the API; it never sees reference routes.

## Run on Laptop 2
    cd src/ui
    npm install
    cp .env.example .env.local     # set VITE_API_URL=http://<laptop1-ip>:8000  (or edit it inside the page)
    npm run dev                    # open http://localhost:5173

Map layers (toggle with the layer control): GPS trace, GPS observations (blue), predicted road segments
(red, labelled MOCK when synthetic), unmatched observations (orange). Map tiles need internet access.
