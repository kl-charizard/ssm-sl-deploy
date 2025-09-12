import UIKit

class ViewController: UIViewController {
    
    @IBOutlet weak var startButton: UIButton!
    @IBOutlet weak var titleLabel: UILabel!
    @IBOutlet weak var descriptionLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }
    
    private func setupUI() {
        titleLabel.text = "Sign Language Detector"
        titleLabel.font = UIFont.boldSystemFont(ofSize: 28)
        titleLabel.textAlignment = .center
        
        descriptionLabel.text = "Real-time American Sign Language recognition using your device's camera"
        descriptionLabel.font = UIFont.systemFont(ofSize: 16)
        descriptionLabel.textAlignment = .center
        descriptionLabel.numberOfLines = 0
        
        startButton.setTitle("Start Detection", for: .normal)
        startButton.backgroundColor = UIColor.systemBlue
        startButton.setTitleColor(.white, for: .normal)
        startButton.layer.cornerRadius = 8
        startButton.titleLabel?.font = UIFont.boldSystemFont(ofSize: 18)
    }
    
    @IBAction func startButtonTapped(_ sender: UIButton) {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        if let cameraVC = storyboard.instantiateViewController(withIdentifier: "CameraViewController") as? CameraViewController {
            cameraVC.modalPresentationStyle = .fullScreen
            present(cameraVC, animated: true)
        }
    }
}
